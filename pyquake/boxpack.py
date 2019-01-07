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

__all__ = (
    'BoxPacker',
)


import random


class BoxPacker:
    def __init__(self, size, central_split_cutoff=64):
        self._central_split_cutoff = central_split_cutoff

        # Attributes that are always set.
        self._size = size

        # Attributes that are set if this is a non-leaf node
        self._children = None
        self._hsplit = None

        # Attributes that are set if this is a leaf containing an image.
        self._image_id = None

    def _make_children(self, hsplit, first_child_size):
        self._hsplit = hsplit
        if hsplit:
            sizes = ((self._size[0], first_child_size),
                     (self._size[0], self._size[1] - first_child_size))
        else:
            sizes = ((first_child_size, self._size[1]),
                     (self._size[0] - first_child_size, self._size[1]))
        self._children = [
            BoxPacker(s, central_split_cutoff=self._central_split_cutoff)
            for s in sizes
        ]

    def insert(self, image_id, size):
        if size[0] > self._size[0] or size[1] > self._size[1]:
            # Can't possibly fit this image in this node.
            return False
        elif self._children:
            # If this node has children, try inserting in them.
            if random.random() < 0.5:
                c = [self._children[0], self._children[1]]
            else:
                c = [self._children[1], self._children[0]]
            for child in c:
                if child.insert(image_id, size):
                    return True
            return False
        elif self._image_id:
            # If this is an image leaf then we can't put another image here.
            return False
        elif size[0] == self._size[0] and size[1] == self._size[1]:
            # Image fits exactly. Just turn the leaf into an image leaf.
            self._image_id = image_id
            return True
        elif (self._size[1] >= self._size[0] and self._size[0] > self._central_split_cutoff and
                size[1] <= self._size[1] // 2):
            # Make a central horizontal split
            self._make_children(hsplit=True, first_child_size=self._size[1] // 2)
            return self.insert(image_id, size)
        elif (self._size[0] >= self._size[1] and self._size[1] > self._central_split_cutoff and
                size[0] <= self._size[0] // 2):
            # Make a central vertical split
            self._make_children(hsplit=False, first_child_size=self._size[0] // 2)
            return self.insert(image_id, size)
        elif self._size[1] - size[1] > self._size[0] - size[0]:
            # Make a horizontal split
            self._make_children(hsplit=True, first_child_size=size[1])
            return self._children[0].insert(image_id, size)
        else:
            # Make a vertical split
            self._make_children(hsplit=False, first_child_size=size[0])
            return self._children[0].insert(image_id, size)

    def __iter__(self):
        if self._children:
            x, y = (0, 0)
            for child in self._children:
                for image_id, (child_x, child_y) in child:
                    yield image_id, (child_x + x, child_y + y)
                if self._hsplit:
                    y += child._size[1]
                else:
                    x += child._size[0]
        else:
            if self._image_id:
                yield (self._image_id, (0, 0))
