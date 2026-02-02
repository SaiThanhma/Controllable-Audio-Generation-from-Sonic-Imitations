from frechet_audio_distance import CLAPScore

clap = CLAPScore(
    submodel_name="630k-audioset",
    enable_fusion=True,
    verbose=True,
)

score = clap.score(
    audio_dir="data/fad_eval_10k",
    text_path="data/prompts.csv",
    text_column="text",
)

print("Mean CLAP similarity:", score)