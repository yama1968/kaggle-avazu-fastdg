N=./
T=$(N)/.
D=$(N)/.
C=/tmp
W=.
M=.
L=./log
U=.
F=.
FG4_5=time pypy $(F)/fastdg_v4_5.py
FG_LAST=$(FG4_5)

#######################################################################################################

%.q02db.fg4.model.gz: $(D)/%.train.csv
	$(FG4_5) train \
	--n_epochs 1 --daily_device_counters --bits 17 \
	--learner_type adpredictor  \
	--group_hours 3 \
	--hourly_counters \
	--interactions \
	--nodayfeature \
	--user_app_site \
	--ad_beta 40 --ad_epsilon 0. \
	--train $< -o $@ 2>&1  | tee $(L)/$@.`date "+%F-%T"`.log

%.g02a.fg4.model.gz: $(D)/%.train.csv
	$(FG4_5) train \
	--interactions \
	--n_epochs 2 --daily_device_counters --L1 0.01 --L2 2.0 --alpha 0.01 --beta 2 --bits 17 \
	--dropout 0.8 \
	--group_hours 2 \
	--hourly_counters \
	--limit_values limit_device_5.json \
	--delayed_learning_factor 0.8 \
	--train $< -o $@ 2>&1  | tee $(L)/$@.`date "+%F-%T"`.log


%.fg4.kaggle.gz: %.fg4.model.gz $(T)/test.gz
	$(FG_LAST) predict --test $(T)/test.gz -i $< -p $@

%.test: %.fg4.model.gz
	$(FG_LAST) predict --test small3.train.csv -i $< -p $@
