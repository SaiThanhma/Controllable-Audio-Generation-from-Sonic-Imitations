from frechet_audio_distance import FrechetAudioDistance

FAD_REF = "/home/stud/dco/Desktop/sketch2sound/eval/generated3"
FAD_EVAL = "/home/stud/dco/Desktop/sketch2sound/eval/generated3"

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
print("FAD score (vggish): %.8f" % score)

fad_clap = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="630k-audioset",  # for CLAP only
    verbose=False,
    enable_fusion=False,            # for CLAP only
)

score = fad_clap.score(
    background_dir=FAD_REF, 
    eval_dir=FAD_EVAL,
    background_embds_path=None,
    eval_embds_path=None
    )

print("FAD score (clap): %.8f" % score)