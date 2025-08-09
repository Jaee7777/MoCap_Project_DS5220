url_CMUMoCap = https://mocap.cs.cmu.edu/allasfamc.zip
url_amc2bvh = https://github.com/thcopeland/amc2bvh/releases/download/v0.1.0/amc2bvh-0.1.0_x86_64_linux.tar
amc2bvh = amc2bvh-0.1.0_x86_64_linux.tar
data_CMU = allasfamc.zip
amc2bvh_path = amc2bvh-0.1.0_x86_64_linux/amc2bvh
data_dir = data/all_asfamc/subjects/
data_dir_01 = data/all_asfamc/subjects/01/

install_amc2bvh:
	curl -LO $(url_amc2bvh);
	tar -xvf $(amc2bvh);
	rm -rf $(amc2bvh)

data_download:
	mkdir data
	cd data; curl -LO $(url_CMUMoCap);
	unzip $(data_CMU)
	rm -rf $(data_CMU)

# Testing with dataset in /01.
data_convert_bvh_01:
	@echo "Converting files..."
	@for file in $(data_dir)01/*.amc; do \
		target=$${file%.amc}$(.bvh); \
		echo "Converting $$file -> $$target.bvh"; \
		$(amc2bvh_path) $(data_dir)01/01.asf $$file -o $$target.bvh; \
	done
	@echo "Converting Done!"

data_convert_csv_01:
	@echo "Converting files..."
	@for file in $(data_dir)01/*.amc; do \
		target=$${file%.amc}$(.bvh); \
		echo "Converting $$target.bvh -> $$target_pos.csv"; \
		bvh2csv --position $$target.bvh; \
	done
	@echo "Converting Done!"

data_generate_3D_01:
	@echo "Merging .csv files..."
	python3 -B src/generate_raw_data.py $(data_dir_01)01_01_pos.csv \
	$(data_dir_01)01_02_pos.csv \
	$(data_dir_01)01_03_pos.csv \
	$(data_dir_01)01_04_pos.csv \
	$(data_dir_01)01_05_pos.csv \
	$(data_dir_01)01_06_pos.csv \
	$(data_dir_01)01_07_pos.csv \
	$(data_dir_01)01_08_pos.csv \
	$(data_dir_01)01_09_pos.csv \
	$(data_dir_01)01_10_pos.csv \
	$(data_dir_01)01_11_pos.csv \
	$(data_dir_01)01_12_pos.csv \
	$(data_dir_01)01_13_pos.csv \
	$(data_dir_01)01_14_pos.csv \
	data/data_CMU_3d_01.csv;
	@echo "Merging Done!"

data_generate_2D_01:
	@echo "Generating 2D csv files..."
	python3 -B src/generate_2d_data.py data/data_CMU_3d_01.csv data/data_CMU_2d_01.csv 0.022119536995887756 -0.022495778277516365 0.013239659368991852
	@echo "Created an input 2D csv file!"

# Using entier dataset.
SUBDIRS := $(shell find $(data_dir) -mindepth 1 -type d)
data_convert_bvh:
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

data_convert_csv:
	@for dir in $(SUBDIRS); do \
		echo "Processing directory: $$dir"; \
		for file in $$dir/*.bvh; do \
			target=$${file%.bvh}$(.bvh); \
			echo "Converting $$file to .csv file"; \
			bvh2csv --position $$file; \
		done; \
	done
	@echo "Converting Done!"


CSV_FILES := $(shell find $$data_dir -name "*_pos.csv" -type f)
data_generate_3D:
	echo "$(CSV_FILES)"
	echo "Merging .csv files..."
	python3 -B src/generate_raw_data.py $(CSV_FILES) data/data_CMU_3d.csv
	echo "Merging Done!"

data_generate_2D:
	@echo "Generating 2D csv files..."
	python3 -B src/generate_2d_data.py data/data_CMU_3d.csv data/data_CMU_2d.csv
	@echo "Created an input 2D csv file!"

plot:
	python3 -B src/csv_animation.py
			$(amc2bvh_path) $(dir)/%.asf $$file -o $$target.bvh; \

dir_trained_model:
	mkdir trained_model/

regression:
	python3 -B src/regression.py

train_full_model:
	python3 -B src/train_full_model.py

start_mocap:
	python3 -B src/mocap_webcam.py