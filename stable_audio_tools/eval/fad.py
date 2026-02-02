from frechet_audio_distance import FrechetAudioDistance

FAD_REF = "/home/stud/dco/Desktop/sketch2sound/eval/generated"
FAD_EVAL = "/home/stud/dco/Desktop/sketch2sound/eval/generated"

fad_vggish = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    audio_load_worker=4,
    verbose=True,
)
score = fad_vggish.score(
    background_dir=FAD_REF, 
    eval_dir=FAD_EVAL,
    background_embds_path=None,
    eval_embds_path=None
    )
print("FAD (CLAP):", score)

# fad_clap = FrechetAudioDistance(
#     model_name="clap",
#     submodel_name="630k-audioset",
#     sample_rate=48000,
#     enable_fusion=True,
#     audio_load_worker=4,
#     verbose=True,
# )

# score = fad_vggish.score(FAD_REF, FAD_EVAL)
# print("FAD (VGGish):", score)