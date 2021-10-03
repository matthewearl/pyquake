"""Tokenizer, as per Quake 3's COM_Parse"""


__all__ = (
    'Token',
    'Tokenizer',
)


import dataclasses
import re


def _try_consume(r, s):
    m = re.search(r, s)
    if m:
        assert m.start(1) == 0
        n = m.end(1)
        t, s = s[:n], s[n:]
    else:
        t = None
    return t, s


@dataclasses.dataclass
class Token:
    s: str
    line_num: int


def _tokenize(s):
    skip_res = [
        r'^(//[^\n]*)(\n|$)',  # line comment
        r'^(/\*(\*[^/]|[^*])*\*/)',  # block comment
        r'^(\s+)(\S|$)',  # white space
    ]

    token_res = [
        r'^("[^"]*")',  # quoted string
        r'^([^"\s]\S*)(\s|$)',  # plain token
    ]

    line_num = 1
    while s:
        while True:
            for skip_re in skip_res:
                t, s = _try_consume(skip_re, s)
                if t is not None:
                    line_num += t.count('\n')
                    break
            else:
                break

        for token_re in token_res:
            t, s = _try_consume(token_re, s)
            if t is not None:
                if t.startswith('"'):
                    assert t.endswith('"')
                    t = t[1:-1]
                yield Token(t, line_num)
                line_num += t.count('\n')


class Tokenizer:
    def __init__(self, s):
        self._iter = iter(_tokenize(s))
        self._peek = []
        self.line_num = 1

    def peek(self, n):
        if n < 1:
            raise ValueError('Cannot peek into past')
        while len(self._peek) < n:
            token = next(self._iter)
            self._peek.append(token)
        return self._peek[n - 1]

    def has(self, n):
        try:
            self.peek(n)
        except StopIteration:
            return False
        else:
            return True

    def __next__(self):
        if self._peek:
            out, self._peek = self._peek[0], self._peek[1:]
        else:
            out = next(self._iter)
        self.line_num = out.line_num

        return out

