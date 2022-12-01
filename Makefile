DATA_DIR := /data
SOURCE_DIR := $(DATA_DIR)/sources
TARGET_DIR := $(DATA_DIR)/targets
PYTHON_DIR := src/bert_coref
NOCOREF_FULLMATCH := examples/pos_nocoref_fullmatch.txt
NOCOREF_PARTMATCH := examples/pos_nocoref_partmatch.txt


all: $(TARGET_DIR)/train.jsonl $(TARGET_DIR)/dev.jsonl $(TARGET_DIR)/test.jsonl

# Save the test data with unaltered labels in order to evaluate POS-based filtering.
# `tail -n +2` is used to skip headers.
$(TARGET_DIR)/test_orig.xtsv:
	@if ! [ -d $(TARGET_DIR) ]; then mkdir $(TARGET_DIR); fi
	@for file in $(SOURCE_DIR)/test/*.xtsv; do \
		python3 $(PYTHON_DIR)/group_sentences.py $${file} | \
		tail -n +2 >> $@; done

$(TARGET_DIR)/test.xtsv: $(TARGET_DIR)/test_orig.xtsv
	@python3 $(PYTHON_DIR)/relabel.py $(TARGET_DIR)/test_orig.xtsv $@ \
		--tag-stop-list $(NOCOREF_FULLMATCH) --subtag-stop-list $(NOCOREF_PARTMATCH) 

eval_filter: $(TARGET_DIR)/test.xtsv
	@paste $(TARGET_DIR)/test_orig.xtsv $(TARGET_DIR)/test.xtsv | \
	cut -d'	' -f2,13 | scripts_utils/precision_recall.awk
.PHONY: eval_filter

$(TARGET_DIR)/train.xtsv $(TARGET_DIR)/dev.xtsv:
	@if ! [ -d $(TARGET_DIR) ]; then mkdir $(TARGET_DIR); fi
	@split="$$(basename $@)"; split=$${split%.*}; \
	for file in $(SOURCE_DIR)/$${split}/*.xtsv; do \
		python3 $(PYTHON_DIR)/group_sentences.py $${file} - | tail -n +2 | \
		python3 $(PYTHON_DIR)/relabel.py --tag-stop-list $(NOCOREF_FULLMATCH) \
			--subtag-stop-list $(NOCOREF_PARTMATCH) \
		>> $(TARGET_DIR)/$${split}.xtsv; done

$(TARGET_DIR)/train.jsonl $(TARGET_DIR)/dev.jsonl $(TARGET_DIR)/test.jsonl: \
		$(TARGET_DIR)/train.xtsv $(TARGET_DIR)/dev.xtsv $(TARGET_DIR)/test.xtsv
	@python3 $(PYTHON_DIR)/conll2jsonl.py $(subst jsonl,xtsv,$@) $@ --create-triplets

$(TARGET_DIR)/baseline.xtsv: $(TARGET_DIR)/test_orig.xtsv
	@echo "form koref anas lemma xpostag upostag feats id deprel head cons" | \
	tr ' ' '\t' >$(TARGET_DIR)/eval_tmp.xtsv; \
	cat $(TARGET_DIR)/test_orig.xtsv >> $(TARGET_DIR)/eval_tmp.xtsv
	@python3 $(PYTHON_DIR)/baseline.py $(TARGET_DIR)/eval_tmp.xtsv | \
	tail -n +2 >$@
	@rm $(TARGET_DIR)/eval_tmp.xtsv

eval_baseline: $(TARGET_DIR)/baseline.xtsv
	@paste $(TARGET_DIR)/test_orig.xtsv $(TARGET_DIR)/baseline.xtsv | cut -d'	' -f2,13 | \
	python3 $(PYTHON_DIR)/clustering/evaluate_clustering.py --metrics purity rand_index NMI
.PHONY: eval_baseline
