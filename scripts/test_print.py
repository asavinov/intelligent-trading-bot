import time
import sys

print('TEST_PRINT: start', flush=True)
for i in range(3):
    print(f'line {i}', flush=True)
    time.sleep(0.2)
print('TEST_PRINT: end', flush=True)
