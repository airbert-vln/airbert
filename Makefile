all :
	r2r speaker
.PHONY : all

help :           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


r2r : 		## Download data for R2R fine-tuning
	python scripts/download.py --beamsearch --config --connectivity --distances --task

speaker : 	## Download data for R2R fine-tuning along with speaker data augmentation
	python scripts/download.py --speaker

few_shot: 	## Build dataset for few-shot learning
	python scripts/few_shot.py

lmdb:           ## Download VLN-BERT faster rcnn LMDB
	mkdir -p data && \
	wget https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip -P data && \
	cd data && \
	unzip matterport-ResNet-101-faster-rcnn-genome.lmdb
