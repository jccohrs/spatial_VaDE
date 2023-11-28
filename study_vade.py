import numpy as np
import train_vade

# seed segmentation
hyperseed = 1337
rng = np.random.default_rng(hyperseed)
low = 0
high = 1e6
seeds = rng.integers(low=low, high=high, size=1)

def create_study_folder():
    directory =f'{args["image_file"]}'
    if args['patches'] and not args['regularizer'] == 0:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/patch-tv')
    elif args['patches']:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/patch')
    elif not args['regularizer'] == 0:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/tv')
    else:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/pure')
    final_directory = os.path.join(current_directory+'/', directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

args = train_vade.read_args()
for s in seeds:
    train_vade.main(s, args)                                      