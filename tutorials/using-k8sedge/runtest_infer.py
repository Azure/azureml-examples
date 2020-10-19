#
# Run this like so:
# 
# $ python runtest_infer.py
#

import requests

#downloading labels for imagenet that resnet model was trained on
classes_entries = requests.get("https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt").text.splitlines()

test_sample = open('snowleopardgaze.jpg', 'rb').read()
print(f"test_sample size is {len(test_sample)}")

try:
    #eg http://51.141.178.47:5001/score
    scoring_uri = 'http://<replace with yout edge device ip address>:5001/score'
    print(f"scoring_uri is {scoring_uri}")

    # Set the content type
    headers = {'Content-Type': 'application/json'}

    # Make the request
    resp = requests.post(scoring_uri, test_sample, headers=headers)

    print("Found a ::" + classes_entries[int(resp.text.strip("[]")) - 1] )

except KeyError as e:
    print(str(e))