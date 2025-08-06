# DS 5220 Summer Project

## Setup

* This repo is based on Linux environment. If you are using Windows, you might have to setup dependencies on your own instead of using the environment.yml file in here. Other Linux commands should be able to work if you are using Anaconda Prompt, which shares most of the regular Linux commands. If not, you simply have to download and/or install some of the tools and convert dataset separately on Windows version instead of using the Make commands.

* Download and unzip [CMU MoCap](https://mocap.cs.cmu.edu/faqs.php) data by following command:
```
make data_download
```


* CMU MoCap data is kept in AMC format. We will convert this into BVH format using [amc2bvh](https://github.com/thcopeland/amc2bvh). Follow the installation guide from [amc2bvh](https://github.com/thcopeland/amc2bvh). The below command can be used on linux terminal to download the .tar file and unzip it to a binary file as in the installation guide:

```
install_amc2bvh
```

* Then, following command can be used to convert amc/asf files into bvh files in the first folder '01.' Use it to make sure everything is working fine:
```
data_convert_01
```

* In order to convert all of thet amc/asf files into bvh files, use following command:
```
data_convert_bvh_all
```

* Now .bvh files can be converted to .csv files using [bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox). This process may take a long time. This can be done by following command:
```
data_convert_csv_all
```


* If any of above doesn't work any more, it might be due to change in [CMU MoCap](https://mocap.cs.cmu.edu/) and/or [amc2bvh](https://github.com/thcopeland/amc2bvh). Please follow their links to check any changes.


## 