curl -X POST -F 'files[]=@peacock-pic.jpg' -F 'files[]=@peacock-pic.jpg' localhost:5001/score 


curl -X POST -F 'data=@peacock-pic.jpg' localhost:5001/score 

# http://aka.ms/peacock-pic 
# $ curl -F ‘data=@path/to/local/file’ UPLOAD_ADDRESS
azmlinfsrv --entry_script code/score-image-size.py --model_dir=models