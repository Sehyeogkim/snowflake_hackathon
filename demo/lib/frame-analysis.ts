export type VideoObservation = {
  image: string;
  time: number;
  motion: number;
};

const EVIDENCE_COUNT = 3;

/**
 * Pick one high-motion observation from each temporal third of the clip.
 * The result preserves before / transition / after evidence without relying
 * on a fixed list of demo-specific frames.
 */
export function selectCriticalFrames(observations: VideoObservation[]): number[] {
  if (observations.length === 0) return [];

  const selections: number[] = [];
  for (let band = 0; band < EVIDENCE_COUNT; band += 1) {
    const start = Math.floor((band * observations.length) / EVIDENCE_COUNT);
    const end = Math.max(
      start + 1,
      Math.floor(((band + 1) * observations.length) / EVIDENCE_COUNT),
    );

    let bestIndex = start;
    for (let index = start + 1; index < end; index += 1) {
      if (observations[index].motion > observations[bestIndex].motion) {
        bestIndex = index;
      }
    }
    selections.push(bestIndex);
  }

  return selections;
}

export function meanAbsoluteDifference(
  previous: Uint8Array | null,
  current: Uint8Array,
): number {
  if (!previous || previous.length !== current.length || current.length === 0) {
    return 0;
  }

  let difference = 0;
  for (let index = 0; index < current.length; index += 1) {
    difference += Math.abs(current[index] - previous[index]);
  }
  return difference / (current.length * 255);
}
