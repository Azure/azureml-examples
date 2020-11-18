 #!/bin/bash

kubectl expose deployment my-infer --type=LoadBalancer --name=my-service-infer
