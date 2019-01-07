# Copyright (c) 2018 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import scipy.interpolate


def _path_distances(path):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


class ProgressMap:

    def __init__(self, reference_positions, num_segments):
        # Make ref a rough approximation of the input paths.  The approximation is constructed by splitting the input
        # into equal length segments.
        distances = _path_distances(reference_positions)

        # Drop points where there wasn't much change in distance (these tend to make interp1d unhappy).
        mask = np.concatenate([np.diff(distances) >= 0.000001, [True]])
        distances = distances[mask]
        reference_positions = reference_positions[mask]
        
        # Normalize distances and create an interpolation function, and then sample this interpolation function at
        # regular intervals.
        distances /= distances[-1]
        f = scipy.interpolate.interp1d(distances, reference_positions, axis=0)
        self._ref = np.array([f(x) for x in np.arange(0, num_segments + 1) / num_segments])

    def get_progress(self, origins):
        # segment_dirs.shape == (num_segments, 3)
        # segment_lengths.shape == (num_segments,)
        segment_dirs = np.diff(self._ref, axis=0)
        segment_lengths = np.linalg.norm(segment_dirs, axis=1)
        segment_dirs /= segment_lengths[:, np.newaxis]

        # Work out the closest point on each line segment
        # segment_dists.shape == active_segments.shape == (num_segments, len(origins))
        # segment_closest_points.shape == (num_segmens, len(origins), 3)
        segment_dists = np.sum(segment_dirs[:, np.newaxis, :] *
                               (origins[np.newaxis, :, :] - self._ref[:-1, np.newaxis, :]),
                               axis=2)
        active_segments = (segment_dists < segment_lengths[:, np.newaxis]) & (segment_dists > 0)
        segment_closest_points = (self._ref[:-1, np.newaxis, :] +
                                    segment_dirs[:, np.newaxis, :] *
                                    segment_dists[:, :, np.newaxis])

        # Calculate vectors which give the distance from the start of the reference path to the closest points, and also
        # to the reference points.
        # point_progress.shape = (len(reference_positions),)
        point_progress = _path_distances(self._ref)
        segment_progress = segment_dists + point_progress[:-1, np.newaxis]

        # Combine the closest points on each segment with the reference points.  Create a validity mask which masks out
        # the inactive segments (ie. those where the closest point does not lie on the segment).
        candidates = np.concatenate([segment_closest_points,
                                     np.broadcast_to(self._ref[:, np.newaxis, :],
                                                     (len(self._ref),) + segment_closest_points.shape[1:])])
        progress = np.concatenate([segment_progress,
                                   np.broadcast_to(point_progress[:, np.newaxis],
                                                   (len(self._ref),) + segment_progress.shape[1:])])
        point_mask = np.concatenate([active_segments,
                                     np.broadcast_to(np.full((len(self._ref), 1), True),
                                                             (len(self._ref),) + active_segments.shape[1:])])
                                     
        # Find the distance between each input origin and candidate point.
        origin_candidate_dist = np.linalg.norm(origins[np.newaxis, :, :] - candidates, axis=2)
        origin_candidate_dist[~point_mask] = np.inf

        min_idx = np.argmin(origin_candidate_dist, axis=0)
        return candidates[min_idx, range(len(origins)), :], progress[min_idx, range(len(origins))]
