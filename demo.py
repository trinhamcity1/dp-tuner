from dp_tuner import main

if __name__ == "__main__":

    main(
        b=256,
        c=1.0,
        epochs=120,
        delta=None,
        sigma_grid=[1.6, 1.8, 2.0],
        seeds=[0, 1],
        gen_kind="dp_ctgan",
        external_data_source=False,
        external_data_path="raw_data\MIMIC_IV_Trasncript.csv"
        )