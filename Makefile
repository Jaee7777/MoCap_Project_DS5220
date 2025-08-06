url_CMUMoCap = https://mocap.cs.cmu.edu/allasfamc.zip
url_amc2bvh = https://github.com/thcopeland/amc2bvh/releases/download/v0.1.0/amc2bvh-0.1.0_x86_64_linux.tar
amc2bvh = amc2bvh-0.1.0_x86_64_linux.tar
data_CMU = allasfamc.zip
amc2bvh_path = amc2bvh-0.1.0_x86_64_linux/amc2bvh
data_dir = data/all_asfamc/subjects/

install_amc2bvh:
	curl -LO $(url_amc2bvh);
	tar -xvf $(amc2bvh);
	rm -rf $(amc2bvh)

data_download:
	mkdir data
	cd data; curl -LO $(url_CMUMoCap);
	unzip $(data_CMU)
	rm -rf $(data_CMU)

data_convert_01:
	@echo "Converting files..."
	@for file in $(data_dir)01/*.amc; do \
		target=$${file%.amc}$(.bvh); \
		echo "Converting $$file -> $$target"; \
		$(amc2bvh_path) $(data_dir)01/01.asf $$file -o $$target.bvh; \
	done
	@echo "Converting Done!"

SUBDIRS := $(shell find $(data_dir) -type d -mindepth 1)
data_convert_bvh_all:
	@for dir in $(SUBDIRS); do \
		echo "Processing directory: $$dir"; \
		for asf_file in $$dir/*.asf; do \
			for file in $$dir/*.amc; do \
				target=$${file%.amc}$(.bvh); \
				echo "Converting $$asf_file $$afile -> $$target.bvh"; \
				$(amc2bvh_path) $$asf_file $$file -o $$target.bvh; \
			done; \
		done; \
	done
	@echo "Converting Done!"

data_convert_csv_all:
	@for dir in $(SUBDIRS); do \
		echo "Processing directory: $$dir"; \
		for file in $$dir/*.bvh; do \
			target=$${file%.bvh}$(.bvh); \
			echo "Converting $$file to .csv file"; \
			bvh2csv --position $$file; \
		done; \
	done
	@echo "Converting Done!"

plot:
	python3 -B src/csv_animation.py
			$(amc2bvh_path) $(dir)/%.asf $$file -o $$target.bvh; \

regression:
	python3 -B src/regression.py