'''
This module implements safety specification class
Dung Tran: Dec/2017
'''


class SafetySpecification(object):
    'Safety Specification Class'

    def __init__(self):
        self.u1 = None
        self.u2 = None
        self.x_range = None
        self.t_range = None

    def set_constraints(self, u1, u2, x_range, t_range):
        'set constraint for safety specification'

        # safety specification has the form of:
        # u1 <= u(x,t) <= u2 for 0 <= x1 <= x <= x2 <= L
        # and 0 <= t1 <= t <= t2 <= T < \infinity

        assert u1 is None or isinstance(u1, float), 'invalid u1'
        assert u2 is None or isinstance(u2, float), 'invalide u2'

        if u1 is not None and u2 is not None:
            assert u1 < u2, 'incorrect constraint, u1 should be smaller than u2'

        assert isinstance(x_range, list) and len(x_range) == 2 and 0 <= x_range[0] <= x_range[1], 'invalid x_range'
        assert isinstance(t_range, list) and len(t_range) == 2 and 0 <= t_range[0] <= t_range[1], 'invalid x_range'

        self.u1 = u1
        self.u2 = u2
        self.x_range = x_range
        self.t_range = t_range
