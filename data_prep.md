## Guide on preparing metadata and audio files

Please first refer to the data download links to download the corresponding audio files and their labels.

We provide template metadata files in the `data_og` folder. These template files typically contains user ID, file_path, label, and split info. However, since we are not allowed to redistribute the label information, we leave the column values as  `NA`. You can use the file_path to locate the corresponding label of each audio file from the data you downloaded. In the end, the metadata file should look something like:

| Uid       | split | voice-path-new             | label |
| --------- | ----- | -------------------------- | ----- |
| devel_001 | 2     | data_og/NCSC/devel_001.wav | NA    |
| devel_002 | 2     | data_og/NCSC/devel_002.wav | NA    |

You can then extract audio files from the audio archives you downloaded into the `data_og` folder, eventually, the folder structure should look like following:

```
data_og/
    Cambridge-EN-metadata.csv
    Cambridge-TASK1-metadata.csv
    DiCOVA2-metadata.csv
    TORGO-metadata.csv
    Nemours-metadata.csv
    NCSC-metadata.csv
    Cambridge/
    DiCOVA2/
    TORGO/
    Nemours/
    NCSC/
```

where each dataset folder would contain the files extracted from the audio archive files. During extraction, keep the internal folder structure intact, as their paths have been given in the existing metadata files. If you would like to modify the folder structure or rename audio files, make sure the metadata column headers are kept the same, and update the `voice-path-new` correspondingly.
