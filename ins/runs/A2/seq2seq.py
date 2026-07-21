"""Char-level GRU seq2seq with attention (torch CPU). Real-ML transduction model.

Trained on fold-train (lang-tagged src -> rep) pairs. NO synthetic augmentation.
Used by the transducer as (a) fallback generator for unseen structures,
(b) a scorer: .score(lang, src, candidate) -> mean logprob, to rerank template variants.
Handles deletions naturally (src -> "" pairs are in training).
"""
import math, random
import torch
import torch.nn as nn

torch.manual_seed(0); random.seed(0)
torch.set_num_threads(5)

PAD, SOS, EOS, UNK = 0, 1, 2, 3


class Vocab:
    def __init__(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def enc(self, s, add_eos=True):
        ids = [self.stoi.get(c, UNK) for c in s]
        if add_eos:
            ids.append(EOS)
        return ids

    def __len__(self):
        return len(self.itos)


class Enc(nn.Module):
    def __init__(self, V, emb=64, hid=96):
        super().__init__()
        self.e = nn.Embedding(V, emb, padding_idx=PAD)
        self.g = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.red = nn.Linear(2 * hid, hid)

    def forward(self, x, lens):
        e = self.e(x)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.g(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h = torch.tanh(self.red(torch.cat([h[0], h[1]], dim=1)))  # (B,hid)
        return out, h  # out (B,T,2hid)


class Attn(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid + 2 * hid, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        T = enc_out.size(1)
        d = dec_h.unsqueeze(1).expand(-1, T, -1)
        e = self.v(torch.tanh(self.W(torch.cat([d, enc_out], dim=2)))).squeeze(2)
        e = e.masked_fill(~mask, -1e9)
        a = torch.softmax(e, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return ctx


class Dec(nn.Module):
    def __init__(self, V, emb=64, hid=96):
        super().__init__()
        self.e = nn.Embedding(V, emb, padding_idx=PAD)
        self.attn = Attn(hid)
        self.g = nn.GRUCell(emb + 2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid, V)

    def step(self, y, h, enc_out, mask):
        ey = self.e(y)
        ctx = self.attn(h, enc_out, mask)
        h = self.g(torch.cat([ey, ctx], dim=1), h)
        logit = self.out(torch.cat([h, ctx], dim=1))
        return logit, h


class Seq2SeqTransducer:
    def __init__(self, hid=96, emb=64, epochs=28, lr=2e-3, bs=64):
        self.hid, self.emb, self.epochs, self.lr, self.bs = hid, emb, epochs, lr, bs

    def _pair_texts(self, df):
        pairs = []
        for r in df.itertuples():
            import json
            edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
            for e in edits:
                src = r.text[e["start"]:e["end"]]
                rep = e["replacement"]
                # language marker prepended as a pseudo-char token
                pairs.append(("\x01" + r.language + "\x02" + src, rep))
        return pairs

    def fit(self, df):
        pairs = self._pair_texts(df)
        self.vocab = Vocab([s for s, _ in pairs] + [t for _, t in pairs])
        V = len(self.vocab)
        self.enc = Enc(V, self.emb, self.hid)
        self.dec = Dec(V, self.emb, self.hid)
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        lossf = nn.CrossEntropyLoss(ignore_index=PAD)
        data = [(self.vocab.enc(s), self.vocab.enc(t)) for s, t in pairs]
        for ep in range(self.epochs):
            random.shuffle(data)
            self.enc.train(); self.dec.train()
            for i in range(0, len(data), self.bs):
                batch = data[i:i + self.bs]
                loss = self._batch_loss(batch, lossf)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
        self.enc.eval(); self.dec.eval()
        return self

    def _prep(self, batch):
        srcs = [b[0] for b in batch]
        slen = torch.tensor([len(s) for s in srcs])
        smax = int(slen.max())
        sx = torch.full((len(batch), smax), PAD, dtype=torch.long)
        for j, s in enumerate(srcs):
            sx[j, :len(s)] = torch.tensor(s)
        mask = (sx != PAD)
        return sx, slen, mask

    def _batch_loss(self, batch, lossf):
        sx, slen, mask = self._prep(batch)
        enc_out, h = self.enc(sx, slen)
        tgts = [b[1] for b in batch]
        tmax = max(len(t) for t in tgts)
        ty = torch.full((len(batch), tmax), PAD, dtype=torch.long)
        for j, t in enumerate(tgts):
            ty[j, :len(t)] = torch.tensor(t)
        y = torch.full((len(batch),), SOS, dtype=torch.long)
        total = 0.0
        for step in range(tmax):
            logit, h = self.dec.step(y, h, enc_out, mask)
            total = total + lossf(logit, ty[:, step])
            y = ty[:, step].clamp(min=0)  # teacher forcing (PAD stays PAD)
        return total / tmax

    @torch.no_grad()
    def generate(self, lang, src, max_extra=12):
        s = self.vocab.enc("\x01" + lang + "\x02" + src)
        sx = torch.tensor([s]); slen = torch.tensor([len(s)])
        mask = (sx != PAD)
        enc_out, h = self.enc(sx, slen)
        y = torch.tensor([SOS])
        maxlen = len(src) * 2 + max_extra
        out = []
        for _ in range(maxlen):
            logit, h = self.dec.step(y, h, enc_out, mask)
            nxt = int(logit.argmax(1))
            if nxt == EOS:
                break
            if nxt >= 4:
                out.append(self.vocab.itos[nxt])
            y = torch.tensor([nxt])
        return "".join(out)

    @torch.no_grad()
    def score(self, lang, src, cand):
        """mean per-char logprob of cand given src (higher=better). For reranking."""
        s = self.vocab.enc("\x01" + lang + "\x02" + src)
        sx = torch.tensor([s]); slen = torch.tensor([len(s)])
        mask = (sx != PAD)
        enc_out, h = self.enc(sx, slen)
        t = self.vocab.enc(cand)
        y = torch.tensor([SOS]); lp = 0.0
        for step in range(len(t)):
            logit, h = self.dec.step(y, h, enc_out, mask)
            logp = torch.log_softmax(logit, dim=1)
            lp += float(logp[0, t[step]])
            y = torch.tensor([t[step]])
        return lp / max(len(t), 1)
