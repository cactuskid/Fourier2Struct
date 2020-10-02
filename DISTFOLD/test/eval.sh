#!/bin/bash

../2017-CONFOLD2/src-bin/confold2-v1.0/core.pl -rr ./predictions/1a3aA.topL-LR-dist.CASPRR -ss ./prepare-psicov150-dataset/sspro_ss_sa/1a3aA.ss_sa -o ./aa

../2017-CONFOLD2/src-bin/confold2-v1.0/third-party-programs/TMscore ./aa/stage1/1a3aA.topL-LR-dist.CASPRR_2.pdb ./prepare-psicov150-dataset/pdb-reindexed/1a3aA.pdb

