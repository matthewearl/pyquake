__all__ = (
        'Simplex',
        'AlreadyOptimal',
        'Infeasible',
)


from typing import NamedTuple

import numpy as np


def _one_hot_encode(i, n):
    out = np.zeros((n,))
    out[i] = 1
    return out


def _mesh_data_from_faces(faces):
    verts = list({v for f in faces for v in f})
    vert_to_idx = {v: i for i, v in enumerate(verts)}
    new_faces = [[vert_to_idx[v] for v in f] for f in faces]
    return verts, new_faces


class AlreadyOptimal(Exception):
    pass


class Infeasible(Exception):
    pass


class Simplex(NamedTuple):
    dim: int
    constraints: np.ndarray
    basic_mask: np.ndarray

    @classmethod
    def from_bbox(cls, bbox_mins, bbox_maxs):
        assert len(bbox_mins) == len(bbox_maxs)

        dim = len(bbox_mins)

        min_constraints = np.stack([np.concatenate([_one_hot_encode(i, dim), [-d]]) for i, d in enumerate(bbox_mins)])
        max_constraints = np.stack([np.concatenate([-_one_hot_encode(i, dim), [d]]) for i, d in enumerate(bbox_maxs)])

        constraints = np.concatenate([min_constraints, max_constraints])
        basic_mask = np.concatenate([np.full((dim,), False), np.full((dim,), True)])

        return cls(dim, constraints, basic_mask)

    @property
    def vert_to_world(self):
        M = np.concatenate([self.constraints[~self.basic_mask], _one_hot_encode(self.dim, self.dim + 1)[None, :]])
        return np.linalg.inv(M)

    @property
    def pos(self):
        return self.vert_to_world[:-1, -1]

    def follow_edge(self, free_idx):
        # B are the constraints in the coordinate space of the current non-basic (free) variables.
        B = self.constraints[self.basic_mask] @ self.vert_to_world

        delta = -B[:, -1] / B[:, free_idx]
        delta[delta < 0] = np.inf
        basic_idx = np.argmin(delta)

        # pivot
        new_basic_mask = self.basic_mask.copy()
        new_basic_mask[np.where(~self.basic_mask)[0][free_idx]] = True
        new_basic_mask[np.where(self.basic_mask)[0][basic_idx]] = False
        
        return Simplex(self.dim, self.constraints, new_basic_mask)

    def iterate(self, c):
        # Transform the optimization axis similarly, select an edge to traverse.
        c = c @ self.vert_to_world[:self.dim, :self.dim]
        for free_idx in range(self.dim):
            if c[free_idx] > 0:
                break
        else:
            raise AlreadyOptimal

        return self.follow_edge(free_idx)

    def _find_verts(self, seen):
        verts = []

        vert_faces = frozenset(i for i, b in enumerate(~self.basic_mask) if b)
        if vert_faces not in seen:
            seen = seen.copy()
            seen.add(vert_faces)
            verts.append((vert_faces, self.vert_to_world[:self.dim, self.dim]))
            for free_idx in range(self.dim):
                seen, new_verts = self.follow_edge(free_idx)._find_verts(seen=seen)
                verts.extend(new_verts)

        return seen, verts

    def find_verts(self):
        seen, verts = self._find_verts(set())
        return verts

    def optimize(self, c):
        s = self
        try:
            while True:
                s = s.iterate(c)
        except AlreadyOptimal:
            pass
        return s

    def to_mesh(self):
        vert_pos = dict(self.find_verts())
        verts = set(vert_pos.keys())

        faces = {f for v in verts for f in v}
        sorted_faces = []
        for face in faces:
            face_verts = {v for v in verts if face in v}

            # These variables will give the next edge/vert from the current vert/edge in the face's edge loop.
            edge_to_vert = {}
            vert_to_edge = {}

            for vert in face_verts:
                other = list(vert - {face})

                # Determine which of the two connected edges is clockwise from `vert`.
                face_norm = self.constraints[face, :3]
                edge_vecs = np.cross(self.constraints[other, :3], face_norm)
                if np.dot(np.cross(edge_vecs[0], edge_vecs[1]), face_norm) > 0:
                    other = list(reversed(other))

                vert_to_edge[vert] = frozenset([face, other[0]])
                edge_to_vert[frozenset([face, other[1]])] = vert

            # Follow the verts around to build the sorted vert list.
            sorted_face = []
            if face_verts:
                first_v = v = next(iter(face_verts))
                for _ in range(len(face_verts)):
                    sorted_face.append(v)
                    v = edge_to_vert[vert_to_edge[v]]
                assert v == first_v
                sorted_faces.append(sorted_face)

        # Create a list of verts, and convert faces into lists of indices into that list.
        mesh_verts, mesh_faces = _mesh_data_from_faces(sorted_faces)
        mesh_verts = [vert_pos[v] for v in mesh_verts]
        return mesh_verts, mesh_faces

    def add_constraint(self, p):
        # Find a point which is behind the plane
        s = self
        try:
            while (p @ s.vert_to_world)[s.dim] >= 0:
                s = s.iterate(-p[:self.dim])
        except AlreadyOptimal:
            # If there's not a point behind the plane, then the constraint is redundant
            return self

        # Find a point on the simplex which is in front of the new plane
        s = self
        try:
            while (p @ s.vert_to_world)[s.dim] < 0:
                s = s.iterate(p[:self.dim])
        except AlreadyOptimal:
            raise Infeasible from None

        new_constraints = np.concatenate([s.constraints, p[None, :]])
        new_basic_mask = np.concatenate([s.basic_mask, [True]])
        return Simplex(self.dim, new_constraints, new_basic_mask)
