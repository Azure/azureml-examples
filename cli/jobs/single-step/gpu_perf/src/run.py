import os
import sys
import pkg_resources

print("*** VERSION ***")

print(sys.version_info)

print("*** PACKAGES ***")
dists = [str(d).replace(" ", "==") for d in pkg_resources.working_set]
for i in dists:
    print(i)
