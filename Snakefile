rule subset_numeric:
    input:
        "data/raw/water_pump_set.csv",
        "data/raw/water_pump_labels.csv"
    output:
        "data/interim/water_pump_set_numeric.csv"
    shell:
        "python vantage/data/make_dataset.py {input} {output}"