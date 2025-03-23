from astroflow import single_pulsar_search_file
from astroflow import Config


def main():
    config = Config(
        dm_low=0,
        dm_high=800,
        freq_start=1130,
        freq_end=1340,
        dm_step=1,
        time_downsample=1,
        t_sample=0.5,
    )
    file = "/home/lingh/work/astroflow/tests/FRB180417.fil"
    output_dir = "./ql"
    single_pulsar_search_file(file, output_dir, config)


main()
