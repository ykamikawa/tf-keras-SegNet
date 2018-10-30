import argparse
import pandas as pd

from  model import segnet
from generator import data_gen_small


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(
            description="SegNet LIP dataset")
    parser.add_argument("--save_dir",
            help="output directory")
    parser.add_argument("--train_list",
            help="train list path")
    parser.add_argument("--trainimg_dir",
            help="train image dir path")
    parser.add_argument("--trainmsk_dir",
            help="train mask dir path")
    parser.add_argument("--val_list",
            help="val list path")
    parser.add_argument("--valimg_dir",
            help="val image dir path")
    parser.add_argument("--valmsk_dir",
            help="val mask dir path")
    parser.add_argument("--batch_size", default=10, type=int,
            help="batch size")
    parser.add_argument("--n_epochs", default=10, type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps", default=100, type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps", default=10, type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels", default=20, type=int,
            help="Number of label")
    parser.add_argument("--input_shape", default=(256, 256, 3),
            help="Input images shape")
    parser.add_argument("--kernel", default=3, type=int,
            help="Kernel size")
    parser.add_argument("--pool_size", default=(2, 2),
            help="pooling and unpooling size")
    parser.add_argument("--output_mode", default="softmax", type=str,
            help="output activation")
    parser.add_argument("--loss", default="categorical_crossentropy", type=str,
            help="loss function")
    parser.add_argument("--optimizer", default="adadelta", type=str,
            help="oprimizer")
    args = parser.parse_args()

    return args


def main(args):
    # set the necessary list
    train_list = pd.read_csv(args.train_list, header=None)
    val_list = pd.read_csv(args.val_list, header=None)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    train_gen = data_gen_small(trainimg_dir, trainmsk_dir,
            train_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)
    val_gen = data_gen_small(valimg_dir, valmsk_dir,
            val_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)

    model = segnet(args.input_shape, args.n_labels,
            args.kernel, args.pool_size, args.output_mode)
    print(model.summary())

    model.compile(loss=args.loss,
            optimizer=args.optimizer, metrics=["accuracy"])
    model.fit_generator(train_gen, steps_per_epoch=args.epoch_steps,
            epochs=args.n_epochs, validation_data=val_gen,
            validation_steps=args.val_steps)

    model.save_weights(args.save_dir+str(args.n_epochs)+".hdf5")
    print("sava weight done..")


if __name__ == "__main__":
    args = argparser()
    main(args)
