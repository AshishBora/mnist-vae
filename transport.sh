REMOTE1=abora@microdeep.ece.utexas.edu:/home/abora/mnist-vae/samples
REMOTE2=abora@microdeep.ece.utexas.edu:/home/abora/mnist-vae/models
LOCAL=/Users/ashish/mnist-vae/
rsync -aurz -e 'ssh -p 52617' --progress $REMOTE1 $LOCAL
rsync -aurz -e 'ssh -p 52617' --progress $REMOTE2 $LOCAL
