/** Where LoRA sits among parameter-efficient fine-tuning (PEFT) methods. */

interface Family {
  name: string;
  trains: string;
  where: string;
  inference: string;
  examples: { label: string; href?: string }[];
}

const FAMILIES: Family[] = [
  {
    name: "Adapter-based",
    trains: "Small new layers (a bottleneck: shrink, then expand)",
    where: "Inserted inside each transformer block",
    inference: "Extra layers run on every forward pass",
    examples: [
      { label: "Houlsby et al., 2019", href: "https://arxiv.org/abs/1902.00751" },
      { label: "AdaptFormer (Chen et al., 2022)", href: "https://arxiv.org/abs/2205.13535" },
    ],
  },
  {
    name: "Prompt-based",
    trains: "A few learned input vectors (\"soft prompts\"); no model weights change",
    where: "Prepended to the input, or to every layer's attention",
    inference: "A slightly longer input",
    examples: [
      { label: "Prompt tuning (Lester et al., 2021)", href: "https://arxiv.org/abs/2104.08691" },
      { label: "Prefix tuning (Li & Liang, 2021)", href: "https://arxiv.org/abs/2101.00190" },
      { label: "Visual prompt tuning (Jia et al., 2022)", href: "https://arxiv.org/abs/2203.12119" },
      { label: "Visual Fourier prompt tuning (Zeng et al., 2024)" },
    ],
  },
  {
    name: "LoRA-based (low-rank)",
    trains: "A low-rank update ΔW = BA to existing weight matrices",
    where: "Added to weights that are already there",
    inference: "None once BA is merged into W",
    examples: [
      { label: "LoRA (Hu et al., 2021)", href: "https://arxiv.org/abs/2106.09685" },
      { label: "QLoRA (Dettmers et al., 2023)", href: "https://arxiv.org/abs/2305.14314" },
    ],
  },
  {
    name: "Selective",
    trains: "A chosen subset of existing parameters, e.g. only the biases",
    where: "Nothing is added",
    inference: "None",
    examples: [
      { label: "BitFit (Ben Zaken et al., 2022)", href: "https://arxiv.org/abs/2106.10199" },
    ],
  },
];

export default function PeftFamily() {
  return (
    <section className="card">
      <h2>LoRA is one of several ways to fine-tune cheaply</h2>
      <p className="body">
        Methods that train only a small number of parameters are called{" "}
        <strong>parameter-efficient fine-tuning (PEFT)</strong>. They differ in{" "}
        <em>what</em> they train and <em>where</em> it sits in the model.
      </p>
      <div className="scroll-x">
        <table className="family">
          <thead>
            <tr><th>Family</th><th>What is trained</th><th>Where it sits</th><th>Cost at inference</th><th>Examples</th></tr>
          </thead>
          <tbody>
            {FAMILIES.map((f) => (
              <tr key={f.name} className={f.name.startsWith("LoRA") ? "row-on" : ""}>
                <td><strong>{f.name}</strong></td>
                <td>{f.trains}</td>
                <td>{f.where}</td>
                <td>{f.inference}</td>
                <td>
                  {f.examples.map((e, i) => (
                    <span key={e.label}>
                      {i > 0 && "; "}
                      {e.href ? <a href={e.href} target="_blank" rel="noreferrer">{e.label}</a> : e.label}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="body">
        Fewer trainable parameters does not automatically mean less memory. TinyTL (
        <a href="https://arxiv.org/abs/2007.11622" target="_blank" rel="noreferrer">Cai et al., 2020</a>)
        showed that the stored activations, not the parameters, are often the bottleneck.
        You will see the same effect measured on the QLoRA page.
      </p>
      <div className="callout">
        <strong>Why this lesson only measures LoRA and QLoRA</strong>
        <span>
          They are the most widely used PEFT methods for large language models, they add
          no inference cost once merged, and QLoRA builds directly on LoRA. Adapter and
          prompt methods were not run here, so nothing on this site measures them. A fair
          question to keep in mind: prompt tuning trains even fewer parameters than LoRA
          at r=4. Would it learn this task's <code>col0</code> / <code>"table"</code>{" "}
          convention as well?
        </span>
      </div>
    </section>
  );
}
