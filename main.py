import cogan
import cyclegan


gan = cyclegan.CycleGAN()
gan.train(epochs=200, batch_size=12, sample_interval=10)


gan = cogan.COGAN()
gan.train(epochs=30000, batch_size=32, sample_interval=200)