"""Reproducible, concealed paired assignment for the proposed experiment."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path


CONFIG = {
    "protocol": "spatial-search-joint-signs-endpoint40-v2",
    "pairs": 150,
    "rounds": 24,
    "sellers": 4,
    "distances": [1, 2, 3, 4],
    "value_hundredths": 12000,
    "outside_hundredths": [4900, 5100],
    "price_hundredths": [2000, 12000],
    "opening_fees_hundredths": [100, 400],
    "gammas": [0, 2, 4],
    "gamma_probability_weights": [2, 1, 2],
    "endowment_hundredths": 4000,
    "score_denominator_hundredths": 20000,
    "fixed_fee_euro_cents": 300,
    "bonus_euro_cents": 1200,
    "practice_cells": [[0, 0], [1, 0], [0, 4], [1, 4]],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class Draws:
    """Domain-separated deterministic draws with rejection, not modulo bias."""

    def __init__(self, key):
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("A key must contain exactly 32 bytes")
        self.key = key
        self.used = set()

    def integer(self, label, low, high):
        if label in self.used:
            raise ValueError("Duplicate randomisation domain")
        if high < low:
            raise ValueError("Empty range")
        self.used.add(label)
        width = high - low + 1
        limit = (1 << 256) - ((1 << 256) % width)
        counter = 0
        while True:
            message = canonical(["spatial-search-draw-v1", label, counter])
            value = int.from_bytes(hmac.new(self.key, message, hashlib.sha256).digest(), "big")
            if value < limit:
                return low + value % width
            counter += 1


def friction_from_ticket(ticket, config):
    """Map five equiprobable integer tickets to the registered 40/20/40 law."""
    weights = config["gamma_probability_weights"]
    if type(ticket) is not int or not 0 <= ticket < sum(weights):
        raise ValueError("Friction ticket outside the registered integer support")
    boundary = 0
    for gamma, weight in zip(config["gammas"], weights):
        boundary += weight
        if ticket < boundary:
            return gamma
    raise AssertionError("Validated ticket was not allocated")


def generate(key, config=None):
    config = dict(CONFIG if config is None else config)
    validate_config(config)
    draws = Draws(key)
    pairs = []
    for b in range(1, config["pairs"] + 1):
        label = f"pair/{b}"
        first_rank = draws.integer(label + "/ranking-coin", 0, 1)
        outside = draws.integer(label + "/outside", *config["outside_hundredths"])
        rounds = []
        for j in range(1, config["rounds"] + 1):
            prefix = label + f"/round/{j}"
            c = draws.integer(prefix + "/fee-level", 0, 1)
            ticket = draws.integer(prefix + "/friction-level", 0,
                                   sum(config["gamma_probability_weights"]) - 1)
            g = friction_from_ticket(ticket, config)
            prices = [draws.integer(prefix + f"/seller/{s}/price", *config["price_hundredths"])
                      for s in range(1, config["sellers"] + 1)]
            rounds.append(dict(round=j, cost=c, gamma=g,
                               opening_fee_hundredths=config["opening_fees_hundredths"][c],
                               seller_prices_hundredths=prices))
        practice = []
        for j, (c, g) in enumerate(config["practice_cells"], 1):
            prices = [draws.integer(label + f"/practice/{j}/seller/{s}/price",
                                    *config["price_hundredths"])
                      for s in range(1, config["sellers"] + 1)]
            practice.append(dict(round=j, cost=c, gamma=g,
                                 opening_fee_hundredths=config["opening_fees_hundredths"][c],
                                 seller_prices_hundredths=prices, paid_eligible=False))
        pairs.append(dict(pair_id=f"B{b:04d}", outside_hundredths=outside,
                          slots=[dict(participant_slot=f"P{2*b-1:04d}", ranking=first_rank),
                                 dict(participant_slot=f"P{2*b:04d}", ranking=1-first_rank)],
                          rounds=rounds, practice=practice))
    commitment = hashlib.sha256(b"spatial-search-commitment-v1\0" + key + canonical(config)).hexdigest()
    manifest = dict(config=config, seed_commitment_sha256=commitment, pairs=pairs)
    validate(manifest)
    return manifest


def validate_config(config):
    """This reference book supports the registered economics, not silent edits."""
    if set(config) != set(CONFIG):
        raise ValueError("Unexpected or missing configuration field")
    for name in ("pairs", "rounds"):
        if type(config[name]) is not int or config[name] < (2 if name == "pairs" else 1):
            raise ValueError("Need integer positive rounds and at least two independent pairs")
    for name in set(CONFIG) - {"pairs", "rounds"}:
        if canonical(config[name]) != canonical(CONFIG[name]):
            raise ValueError("Economic configuration differs from the registered reference: " + name)


def validate(manifest):
    c = manifest["config"]
    validate_config(c)
    if len(manifest["pairs"]) != c["pairs"]:
        raise ValueError("Wrong pair count")
    slots = set()
    pair_ids = set()
    for pair in manifest["pairs"]:
        if pair["pair_id"] in pair_ids:
            raise ValueError("Duplicate pair identifier")
        pair_ids.add(pair["pair_id"])
        if sorted(s["ranking"] for s in pair["slots"]) != [0, 1]:
            raise ValueError("Every pair must have one of each ranking")
        if len(pair["rounds"]) != c["rounds"]:
            raise ValueError("Wrong planned round count")
        if (type(pair["outside_hundredths"]) is not int or
                not c["outside_hundredths"][0] <= pair["outside_hundredths"] <= c["outside_hundredths"][1]):
            raise ValueError("Outside option outside support")
        for slot in pair["slots"]:
            if slot["participant_slot"] in slots:
                raise ValueError("Duplicate participant slot")
            slots.add(slot["participant_slot"])
        if len(pair.get("practice", [])) != len(c["practice_cells"]):
            raise ValueError("Wrong practice market count")
        for group in ("rounds", "practice"):
            for j, row in enumerate(pair[group], 1):
                if (type(row["round"]) is not int or row["round"] != j or
                        type(row["cost"]) is not int or row["cost"] not in (0, 1) or
                        type(row["gamma"]) is not int or row["gamma"] not in c["gammas"]):
                    raise ValueError("Invalid treatment record")
                if (type(row["opening_fee_hundredths"]) is not int or
                        row["opening_fee_hundredths"] != c["opening_fees_hundredths"][row["cost"]]):
                    raise ValueError("Fee mismatch")
                if len(row["seller_prices_hundredths"]) != c["sellers"]:
                    raise ValueError("Wrong seller count")
                if any(type(p) is not int or
                       not c["price_hundredths"][0] <= p <= c["price_hundredths"][1]
                       for p in row["seller_prices_hundredths"]):
                    raise ValueError("Price outside integer support")
                if group == "practice":
                    if [row["cost"], row["gamma"]] != c["practice_cells"][j - 1]:
                        raise ValueError("Practice treatment or order mismatch")
                    if row.get("paid_eligible") is not False:
                        raise ValueError("Practice must be unpaid")
    return True


def audit(manifest):
    """Audit realised balance without imposing it or selecting a better book."""
    counts = {f"c{c}_g{g}": 0 for c in (0, 1) for g in (0, 2, 4)}
    absent = 0
    coincident_profiles = 0
    for pair in manifest["pairs"]:
        local = set()
        profiles = set()
        for row in pair["rounds"]:
            cell = f"c{row['cost']}_g{row['gamma']}"
            counts[cell] += 1
            local.add(cell)
            profile = tuple(row["seller_prices_hundredths"])
            coincident_profiles += int(profile in profiles)
            profiles.add(profile)
        absent += int(len(local) < 6)
    return dict(pair_round_counts=counts, pairs_without_all_six_cells=absent,
                chance_repeated_price_vectors_within_pair=coincident_profiles,
                action="No rerandomisation, trimming, balancing or profile screening")


def write_exclusive(path, value):
    payload = json.dumps(value, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo-seed", type=int)
    group.add_argument("--secret-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=CONFIG["pairs"])
    parser.add_argument("--rounds", type=int, default=CONFIG["rounds"])
    args = parser.parse_args()
    demo = args.demo_seed is not None
    key = (hashlib.sha256(f"PUBLIC DEMONSTRATION ONLY {args.demo_seed}".encode()).digest()
           if demo else args.secret_file.read_bytes())
    config = dict(CONFIG, pairs=args.pairs, rounds=args.rounds)
    manifest = generate(key, config)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The private book must remain server-side until collection has ended.
    write_exclusive(args.output_dir / "private_assignment_book.json", manifest)
    write_exclusive(args.output_dir / "commitment.json",
                    dict(config=config, demonstration_only=demo,
                         public_demo_seed=args.demo_seed,
                         seed_commitment_sha256=manifest["seed_commitment_sha256"]))
    write_exclusive(args.output_dir / "assignment_audit.json", audit(manifest))
    print("Demonstration only." if demo else "Operational book generated; keep book and key private.")
    print("Commitment SHA-256:", manifest["seed_commitment_sha256"])
    print("No participants assigned, no site deployed, no payment made.")


if __name__ == "__main__":
    main()
