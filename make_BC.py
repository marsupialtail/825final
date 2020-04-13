import numpy as np
import sys
matrix = np.load(sys.argv[1])
A = matrix.shape[1]
B = matrix.shape[0]
a = np.random.normal(size=(B,int(sys.argv[2])))

if len(sys.argv) > 4:
    in_format = sys.argv[3]
    out_format = sys.argv[4]
else:
    in_format = "NCHW"
    out_format = "NCHW"

fuse = False
if len(sys.argv) > 5:
    bias = np.load(sys.argv[5])
    fuse = True

print(in_format,out_format)

if in_format == "NCHW":
    np.save("BC.npy",a.astype(np.float32))
elif in_format == "NHWC":
    np.save("BC.npy",a.astype(np.float32).transpose().copy())
else:
    print("Unsupported in format")
if not fuse:
    if out_format == "NCHW":
        np.save("ref.npy",np.dot(matrix.transpose(),a).astype(np.float32))
    elif out_format == "NHWC":
        np.save("ref.npy",np.dot(matrix.transpose(),a).astype(np.float32).transpose().copy())
    else:
        print("Unsupported out format")
else:
    if out_format == "NCHW":
        np.save("ref.npy",np.maximum(0,np.dot(matrix.transpose(),a) + np.expand_dims(bias,1)).astype(np.float32))
    elif out_format == "NHWC":
        np.save("ref.npy",np.maximum(0,np.dot(matrix.transpose(),a) + bias).astype(np.float32).transpose().copy())
    else:
        print("Unsupported out format")
