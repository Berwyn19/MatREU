"""
Adapted from Matteo Niccoli's perceptural rainbow (2013).
mycarta.wordpress.com/2013/02/21/perceptual-rainbow-palette-the-method/
"""
from matplotlib import colors


cubehelix_values = [[0.4706, 0.0, 0.5216], [0.4749, 0.0003, 0.5332],
                    [0.4791, 0.001, 0.5447], [0.4831, 0.0022, 0.5562],
                    [0.4869, 0.0039, 0.5676], [0.4905, 0.0059, 0.579],
                    [0.4939, 0.0084, 0.5902], [0.4971, 0.0111, 0.6014],
                    [0.5001, 0.0143, 0.6126], [0.5028, 0.0177, 0.6236],
                    [0.5052, 0.0214, 0.6346], [0.5074, 0.0253, 0.6456],
                    [0.5093, 0.0295, 0.6564], [0.5108, 0.0339, 0.6672],
                    [0.5121, 0.0384, 0.6779], [0.513, 0.0431, 0.6885],
                    [0.5135, 0.0478, 0.6991], [0.5137, 0.0527, 0.7096],
                    [0.5136, 0.0585, 0.7201], [0.5133, 0.0659, 0.7307],
                    [0.5129, 0.0749, 0.7413], [0.5122, 0.0851, 0.7519],
                    [0.5115, 0.0964, 0.7625], [0.5105, 0.1086, 0.773],
                    [0.5095, 0.1216, 0.7835], [0.5083, 0.1351, 0.7939],
                    [0.507, 0.149, 0.8042], [0.5057, 0.1631, 0.8143],
                    [0.5042, 0.1771, 0.8242], [0.5027, 0.1909, 0.8339],
                    [0.5011, 0.2044, 0.8433], [0.4994, 0.2173, 0.8525],
                    [0.4977, 0.2294, 0.8614], [0.496, 0.2406, 0.8699],
                    [0.4942, 0.2507, 0.8781], [0.4922, 0.26, 0.8863],
                    [0.4897, 0.269, 0.8949], [0.4869, 0.2776, 0.9038],
                    [0.4837, 0.2861, 0.9128], [0.4801, 0.2943, 0.9218],
                    [0.4763, 0.3023, 0.9308], [0.4723, 0.3102, 0.9396],
                    [0.4681, 0.3179, 0.9481], [0.4638, 0.3255, 0.9561],
                    [0.4594, 0.333, 0.9637], [0.4549, 0.3405, 0.9706],
                    [0.4504, 0.3479, 0.9767], [0.446, 0.3553, 0.982],
                    [0.4416, 0.3628, 0.9863], [0.4374, 0.3704, 0.9895],
                    [0.4334, 0.378, 0.9915], [0.4296, 0.3858, 0.9922],
                    [0.426, 0.3937, 0.992], [0.4224, 0.4016, 0.9913],
                    [0.4188, 0.4096, 0.9902], [0.4153, 0.4177, 0.9887],
                    [0.4119, 0.4257, 0.9869], [0.4084, 0.4338, 0.9848],
                    [0.405, 0.4418, 0.9824], [0.4015, 0.4498, 0.9797],
                    [0.3981, 0.4578, 0.9768], [0.3946, 0.4656, 0.9737],
                    [0.3911, 0.4734, 0.9705], [0.3876, 0.4811, 0.9671],
                    [0.384, 0.4886, 0.9637], [0.3804, 0.496, 0.9601],
                    [0.3767, 0.5033, 0.9566], [0.3729, 0.5103, 0.953],
                    [0.3691, 0.5172, 0.9495], [0.3652, 0.5239, 0.9458],
                    [0.3612, 0.5304, 0.9417], [0.3571, 0.5368, 0.9372],
                    [0.353, 0.5431, 0.9323], [0.3488, 0.5493, 0.9271],
                    [0.3445, 0.5554, 0.9217], [0.3403, 0.5614, 0.916],
                    [0.3359, 0.5673, 0.91], [0.3316, 0.5732, 0.9039],
                    [0.3272, 0.579, 0.8976], [0.3228, 0.5848, 0.8912],
                    [0.3184, 0.5905, 0.8846], [0.314, 0.5962, 0.878],
                    [0.3095, 0.6019, 0.8714], [0.3051, 0.6076, 0.8647],
                    [0.3007, 0.6134, 0.8581], [0.2963, 0.6191, 0.8515],
                    [0.2917, 0.6249, 0.8449], [0.2865, 0.6307, 0.838],
                    [0.2811, 0.6365, 0.8311], [0.2753, 0.6424, 0.8239],
                    [0.2694, 0.6482, 0.8167], [0.2634, 0.654, 0.8092],
                    [0.2575, 0.6598, 0.8017], [0.2516, 0.6656, 0.7941],
                    [0.246, 0.6713, 0.7864], [0.2407, 0.6769, 0.7786],
                    [0.2357, 0.6825, 0.7707], [0.2313, 0.6879, 0.7627],
                    [0.2274, 0.6933, 0.7548], [0.2243, 0.6985, 0.7467],
                    [0.2219, 0.7036, 0.7387], [0.2204, 0.7086, 0.7306],
                    [0.2199, 0.7134, 0.7225], [0.2202, 0.7181, 0.7144],
                    [0.2211, 0.7227, 0.7061], [0.2224, 0.7272, 0.6978],
                    [0.2242, 0.7317, 0.6893], [0.2264, 0.7361, 0.6808],
                    [0.229, 0.7404, 0.6722], [0.2318, 0.7446, 0.6636],
                    [0.2349, 0.7488, 0.6549], [0.2382, 0.7529, 0.6461],
                    [0.2415, 0.7569, 0.6373], [0.245, 0.7609, 0.6285],
                    [0.2485, 0.7648, 0.6197], [0.2519, 0.7687, 0.6108],
                    [0.2553, 0.7725, 0.602], [0.2585, 0.7762, 0.5932],
                    [0.2615, 0.7799, 0.5844], [0.2643, 0.7836, 0.5756],
                    [0.2669, 0.7872, 0.5668], [0.2695, 0.7907, 0.558],
                    [0.2721, 0.7942, 0.5491], [0.2747, 0.7975, 0.5401],
                    [0.2773, 0.8009, 0.5312], [0.2799, 0.8042, 0.5222],
                    [0.2825, 0.8074, 0.5132], [0.2851, 0.8106, 0.5042],
                    [0.2877, 0.8138, 0.4952], [0.2903, 0.8169, 0.4862],
                    [0.2929, 0.82, 0.4773], [0.2956, 0.8231, 0.4684],
                    [0.2983, 0.8263, 0.4595], [0.301, 0.8294, 0.4507],
                    [0.3038, 0.8325, 0.442], [0.3066, 0.8356, 0.4334],
                    [0.3094, 0.8388, 0.4248], [0.3122, 0.842, 0.4159],
                    [0.3149, 0.8453, 0.4062], [0.3176, 0.8487, 0.396],
                    [0.3202, 0.852, 0.3854], [0.3227, 0.8555, 0.3746],
                    [0.3253, 0.8589, 0.3637], [0.328, 0.8623, 0.3529],
                    [0.3307, 0.8656, 0.3423], [0.3334, 0.8689, 0.3322],
                    [0.3363, 0.8722, 0.3227], [0.3394, 0.8754, 0.3139],
                    [0.3426, 0.8784, 0.306], [0.3461, 0.8814, 0.2991],
                    [0.3497, 0.8842, 0.2935], [0.3536, 0.8869, 0.2894],
                    [0.3578, 0.8894, 0.2867], [0.3623, 0.8917, 0.2858],
                    [0.3675, 0.8939, 0.286], [0.3737, 0.8962, 0.2865],
                    [0.3807, 0.8984, 0.2874], [0.3886, 0.9007, 0.2886],
                    [0.3971, 0.9029, 0.29], [0.4063, 0.905, 0.2917],
                    [0.4159, 0.9071, 0.2935], [0.426, 0.9091, 0.2954],
                    [0.4364, 0.911, 0.2975], [0.4471, 0.9128, 0.2997],
                    [0.4578, 0.9145, 0.3018], [0.4687, 0.916, 0.304],
                    [0.4794, 0.9174, 0.3062], [0.49, 0.9186, 0.3083],
                    [0.5004, 0.9196, 0.3102], [0.5104, 0.9204, 0.3121],
                    [0.52, 0.921, 0.3137], [0.5294, 0.9215, 0.3153],
                    [0.539, 0.9219, 0.3168], [0.5487, 0.9223, 0.3184],
                    [0.5585, 0.9227, 0.32], [0.5683, 0.9231, 0.3216],
                    [0.5781, 0.9234, 0.3232], [0.588, 0.9238, 0.3248],
                    [0.5978, 0.9241, 0.3264], [0.6076, 0.9244, 0.3279],
                    [0.6173, 0.9246, 0.3294], [0.6268, 0.9248, 0.3309],
                    [0.6362, 0.925, 0.3323], [0.6454, 0.9252, 0.3337],
                    [0.6545, 0.9253, 0.335], [0.6633, 0.9254, 0.3363],
                    [0.6718, 0.9255, 0.3375], [0.68, 0.9255, 0.3386],
                    [0.6881, 0.9255, 0.3397], [0.6961, 0.9255, 0.3407],
                    [0.704, 0.9255, 0.3417], [0.7119, 0.9255, 0.3427],
                    [0.7197, 0.9255, 0.3436], [0.7274, 0.9255, 0.3446],
                    [0.735, 0.9255, 0.3455], [0.7424, 0.9255, 0.3463],
                    [0.7496, 0.9255, 0.3472], [0.7567, 0.9255, 0.348],
                    [0.7636, 0.9255, 0.3488], [0.7703, 0.9255, 0.3495],
                    [0.7768, 0.9255, 0.3503], [0.783, 0.9255, 0.351],
                    [0.7889, 0.9255, 0.3516], [0.7946, 0.9255, 0.3523],
                    [0.8, 0.9255, 0.3529], [0.8051, 0.9251, 0.3535],
                    [0.8099, 0.9238, 0.354], [0.8145, 0.9219, 0.3546],
                    [0.8189, 0.9192, 0.3551], [0.8232, 0.916, 0.3556],
                    [0.8272, 0.9122, 0.356], [0.8312, 0.908, 0.3565],
                    [0.8351, 0.9033, 0.3569], [0.8389, 0.8984, 0.3574],
                    [0.8427, 0.8931, 0.3578], [0.8465, 0.8877, 0.3582],
                    [0.8503, 0.8822, 0.3586], [0.8541, 0.8765, 0.359],
                    [0.858, 0.871, 0.3595], [0.8621, 0.8654, 0.3599],
                    [0.8663, 0.8601, 0.3603], [0.8706, 0.8549, 0.3608],
                    [0.8752, 0.8498, 0.3613], [0.8801, 0.8445, 0.3618],
                    [0.8853, 0.839, 0.3624], [0.8907, 0.8334, 0.363],
                    [0.8962, 0.8276, 0.3636], [0.9017, 0.8217, 0.3642],
                    [0.9073, 0.8156, 0.3648], [0.9129, 0.8093, 0.3654],
                    [0.9183, 0.8029, 0.366], [0.9236, 0.7963, 0.3665],
                    [0.9287, 0.7896, 0.367], [0.9335, 0.7828, 0.3675],
                    [0.938, 0.7758, 0.3678], [0.9421, 0.7687, 0.3682],
                    [0.9457, 0.7615, 0.3684], [0.9488, 0.7541, 0.3685],
                    [0.9514, 0.7466, 0.3686], [0.9536, 0.7389, 0.3686],
                    [0.9559, 0.731, 0.3684], [0.9581, 0.7229, 0.3682],
                    [0.9602, 0.7145, 0.3679], [0.9622, 0.706, 0.3675],
                    [0.9642, 0.6972, 0.367], [0.9661, 0.6882, 0.3664],
                    [0.9679, 0.6791, 0.3658], [0.9695, 0.6697, 0.3651],
                    [0.9711, 0.6602, 0.3643], [0.9724, 0.6505, 0.3634],
                    [0.9736, 0.6406, 0.3625], [0.9746, 0.6305, 0.3615],
                    [0.9754, 0.6203, 0.3604], [0.976, 0.6099, 0.3593],
                    [0.9764, 0.5994, 0.3581], [0.9765, 0.5887, 0.3569]]


c_rainbow = colors.ListedColormap(cubehelix_values)