from pygit2 import Repository

try:
    print(Repository(".").head.shorthand)
except:
    print("Repository('.').head.shorthand error")
