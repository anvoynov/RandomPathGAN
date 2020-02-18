import torch
from torch.nn import functional as F


def train_blocks_predictor(rp_generator, blocks_predictor, out_file=None,
                           rate=0.002, steps_per_decay=None, num_steps=10000):
    optimizer = torch.optim.Adam(blocks_predictor.parameters(), lr=rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, steps_per_decay)
    steps_per_log = 100
    batch_size = 64

    for step in range(num_steps):
        with torch.no_grad():
            imgs = rp_generator(batch_size)

        blocks_predictor.zero_grad()
        indices_prediction = blocks_predictor(imgs)

        loss = 0.0
        for i, bucket in enumerate(rp_generator.buckets()):
            true_indices = torch.tensor(bucket.recent_indices, dtype=torch.long, device='cuda')
            loss += F.cross_entropy(indices_prediction[i], true_indices)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step % steps_per_log == 0 and step > 0:
            val_steps = 20
            matches = [0.0 for _ in blocks_predictor.classifiers]
            loss = 0.0
            for _ in range(val_steps):
                with torch.no_grad():
                    imgs = rp_generator(batch_size)
                    indices_prediction = blocks_predictor(imgs)

                    for i, bucket in enumerate(rp_generator.buckets()):
                        true_indices = torch.tensor(
                            bucket.recent_indices, dtype=torch.long, device='cuda')
                        loss += F.cross_entropy(indices_prediction[i], true_indices)
                        matches[i] += torch.mean((torch.argmax(
                            indices_prediction[i], dim=1) == true_indices).to(torch.float))

            loss = loss / val_steps
            matches = [m / val_steps for m in matches]

            matches_str = ''
            for i, match in enumerate(matches):
                matches_str += ' {:.2} |'.format(match)
                matches[i] = 0.0
            print('step {}: {:.2} | matches: {}'.format(step, loss.item(), matches_str))

    if out_file is not None:
        torch.save(blocks_predictor.state_dict(), out_file)

    return blocks_predictor