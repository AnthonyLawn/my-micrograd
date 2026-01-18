import math
import random

from micrograd.engine import Value as KValue
from my_micrograd.engine import Value as MyValue

class TestValueCompatibility:

    def build_graph(self, V):
        x = V(2.0)
        y = V(-3.0)
        z = V(10.0)

        q = x + y
        f = q * z
        out = f.relu() + (x * y)

        return x, y, z, out
    
    def test_forward(self):
        _, _, _, k_out = self.build_graph(KValue)
        _, _, _, my_out = self.build_graph(MyValue)

        assert math.isclose(k_out.data, my_out.data, rel_tol=1e-6)

    def test_backward(self):
        kx, ky, kz, k_out = self.build_graph(KValue)
        myx, myy, myz, my_out = self.build_graph(MyValue)

        k_out.backward()
        my_out.backward()

        assert math.isclose(kx.grad, myx.grad, rel_tol=1e-6)
        assert math.isclose(ky.grad, myy.grad, rel_tol=1e-6)
        assert math.isclose(kz.grad, myz.grad, rel_tol=1e-6)

    def random_ops(self):
        ops = []
        for _ in range(5):
            op = random.choice(['add', 'mul', 'pow', 'relu'])
            ops.append(op)
        return ops

    def build_from_recipe(self, V, ops, x_raw, y_raw):
        x = V(x_raw)
        y = V(y_raw)

        out = x
        for op in ops:
            if op == 'add':
                out = out + y
            elif op == 'mul':
                out = out * y
            elif op == 'pow':
                out = out ** 2
            elif op == 'relu':
                out = out.relu()

        return x, y, out
    
    def test_random_graph(self):
        x_raw = random.uniform(-1, 1)
        y_raw = random.uniform(-1, 1)
        ops = self.random_ops()

        kx, ky, kout = self.build_from_recipe(KValue, ops, x_raw, y_raw)
        myx, myy, myout = self.build_from_recipe(MyValue, ops, x_raw, y_raw)

        kout.backward()
        myout.backward()

        assert math.isclose(kout.data, myout.data, rel_tol=1e-6)
        assert math.isclose(kx.grad, myx.grad, rel_tol=1e-6)
        assert math.isclose(ky.grad, myy.grad, rel_tol=1e-6)