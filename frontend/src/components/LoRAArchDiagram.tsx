export default function LoRAArchDiagram() {
  return (
    <section className="card">
      <h2>The architecture, before the real numbers</h2>
      <p className="body">
        Every forward pass runs two paths over the same input and adds their outputs.
        Only one of the two paths ever receives a gradient.
      </p>
      <div className="arch-frame">
        <img src={`${import.meta.env.BASE_URL}lora-architecture.webp`}
             alt="LoRA architecture: input x feeds both the frozen pretrained weights W0 and a low-rank adapter path (A then B); the two outputs are summed to produce h" />
      </div>
      <p className="muted">
        h = W₀x + BAx — W₀ never moves; A and B are the entire trainable surface,
        multiplied together and added on top.
      </p>
    </section>
  );
}
