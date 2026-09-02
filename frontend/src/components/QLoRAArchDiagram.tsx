export default function QLoRAArchDiagram() {
  return (
    <section className="card">
      <h2>The architecture, before the real numbers</h2>
      <p className="body">
        The same two paths as LoRA, with one change: the frozen weights are quantised
        to 4 bits, and only the low-rank adapters train, same as before.
      </p>
      <div className="arch-frame">
        <img src={`${import.meta.env.BASE_URL}qlora-architecture.webp`}
             alt="QLoRA architecture: input x feeds both the 4-bit quantised weights Wq and a low-rank adapter path (A then B); the two outputs are summed to produce h" />
      </div>
      <p className="muted">
        Nothing about the arithmetic changes from LoRA — only how the frozen operand is
        stored. The adapters never touch 4 bits; they stay in bf16 because they are the
        part actually being trained.
      </p>
    </section>
  );
}
