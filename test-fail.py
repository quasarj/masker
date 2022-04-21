#!/usr/bin/env python3

import sys
from enum import Enum


class Test(Enum):
    A = 1
    B = 2


print("some output here")
print(Test.A.value)
sys.exit(Test.A.value)
sys.exit(13)
