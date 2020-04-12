import torch
import numpy as np


def normalize(vecs, order=None):
    norms = np.linalg.norm(vecs, axis=1, ord=order)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix,
                                out=np.zeros_like(vecs),
                                where=norms_matrix != 0)
    return norms, normalized_vecs


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

class NearestNeighborCompressor(object):
    def __init__(self, size, shape, args, thread=1):
        c_dim = args.c_dim
        k_bit = args.k_bit
        n_bit = args.n_bit

        assert c_dim > 0
        assert k_bit >= 0
        assert n_bit > 0

        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.thread = thread

        if c_dim == 0 or self.size < args.c_dim:
            self.dim = self.size
        else:
            self.dim = c_dim
            for i in range(0, 10):
                if size % self.dim != 0:
                    self.dim = self.dim // 2 * 3

        if c_dim != self.dim:
            print("alternate dimension form"
                  " {} to {}, size {} shape {}"
                  .format(c_dim, self.dim, size, shape))

        assert size % self.dim == 0, \
            "not divisible size/shape {}/{}  c_dim {} self.dim {}"\
                .format(size, shape, c_dim, self.dim)

        if k_bit <= 0:
            self.K = self.dim
        else:
            self.K = 2 ** k_bit

        # self.codewords = np.random.normal(size=(self.K, self.dim))
        # self.codewords = normalize(self.codewords)[1].astype(np.float32)
        location = './codebooks/{}/angular_dim_{}_Ks_{}.fvecs'.format(
            'learned_codebook', self.dim, self.K)
        self.codewords = normalize(fvecs_read(location))[1]
        self.codewords = torch.from_numpy(self.codewords)

        if self.cuda:
            self.codewords = self.codewords.cuda()
        self.code_dtype = torch.uint8 if k_bit <= 8 else torch.int32

        self.compressed_norm = False

    def compress(self, vec):

        vec = vec.view(-1, self.dim)

        # calculate probability, complexity: O(d*K)
        p = torch.mm(self.codewords, vec.transpose(0, 1)).transpose(0, 1)
        probability = torch.abs(p)

        # choose codeword
        codes = torch.argmax(probability, dim=1)
        u = p.gather(dim=1, index=codes.view(-1, 1)).view(-1)

        if self.compressed_norm:
            u = self.norm_compressor.compress(u)

        return [u, codes.type(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature

        codes = codes.view(-1).type(torch.long)
        norms = norms.view(-1)

        vec = self.codewords[codes]
        recover = torch.mul(vec, norms.view(-1, 1).expand_as(vec))
        return recover.view([self.thread]+self.shape).mean(dim=0)
