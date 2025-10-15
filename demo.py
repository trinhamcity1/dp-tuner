from dp_tuner import main

main(
    b=256,
    c=1.0,
    epochs=120,
    delta=None,
    sigma_grid=[1.6, 1.8, 2.0],
    seeds=[0, 1],
    gen_kind="dp_ctgan"
    )