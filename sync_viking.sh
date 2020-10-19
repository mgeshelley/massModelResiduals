cd ~/Documents/phd/machine_learning/massModelResiduals

rsync -avz --include-from=.rsync_include --exclude-from=.rsync_exclude * mges501@viking.york.ac.uk:~/scratch/massModelResiduals