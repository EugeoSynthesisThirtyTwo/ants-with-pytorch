import torch as th

n = 5
m = 3
k = 4

texture = th.randint(0, 5, (n, n, 2))
indices = th.randint(0, n, (m, k, 2))

print(f"{texture=}")
print(f"{indices=}")

print(f"{indices[:, :, 0]=}")
print(f"{indices[:, :, 1]=}")

# texture = (n, n)
# indices = (m, k, 2)
# indices[:, :, 0] = (m, k)
# values = (m, k)
values = texture[indices[:, :, 0], indices[:, :, 1]]
print(f"{values=}")
