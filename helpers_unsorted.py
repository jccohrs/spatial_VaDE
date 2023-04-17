

def classify(output_path, image, weights, gt, n_cl, gt_file, ignore_label=None, add_spatial=False):
    ''' classify an image using a trained VaDE model '''

    im = rescale_intensity(image, out_range=(0.0, 1.0))
    args = {'spatial_coord': add_spatial}
    args['use_mnf'] = False
    args['use_pca'] = False
    args['ignore_label'] = ignore_label
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['batch_size'] = 100
    args['latent_dim'] = 10
    args['n_cl'] = n_cl
    args['regularizer'] = 0.002
    args['gt_file'] = gt_file

    args = prepare_data(im, gt, args, None, None)

    # init model
    vade = VaDE(input_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=im.shape[:-1],
                one_by_delta_x=args['fd_inv_stepsize'], regularization=args['regularizer'],
                directory=output_path, ignore_label=args['ignore_label'])

    print('Initialized VaDE model.')
    vade.to(args['device'])
    print(f'Pushed model to {args["device"]}.')

    #------------------------------------------------------------------------------------#
    # load pretrained weights
    #------------------------------------------------------------------------------------#

    print(f"Load model from {weights}...")
    vade.load_model(weights)
    print('Loaded pretrained network weights.')

    print('Use model to classify the data...')
    loss, _, y_pred, gamma = vade.predict(args['validloader'])

    saveArrayAsNetCDF(gamma.reshape((args['im'].shape[0], args['im'].shape[1], gamma.shape[-1])),
                      f'{output_path}/gamma.nc')
    seg = y_pred.reshape((args['im'].shape[:-1]))

    print(f'Validation loss: {loss / len(args["validloader"])}')
    print(f'Segmentation scores: {segmentation_scores(seg, args["gt"], args["ignore_label_inside"])}')

    save_segmentation(output_path, f'prediction_spat{int(args["spatial_coord"])}', seg, args)

