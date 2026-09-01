import { diffMask, spacer, tokenize } from "../lib/sqldiff";

/** SQL rendered with the tokens that differ from gold marked. */
export default function SqlText({ sql, gold }: { sql: string; gold?: string }) {
  const toks = tokenize(sql);
  const mask = gold ? diffMask(sql, gold) : null;
  return (
    <code className="sql">
      {toks.map((t, i) => (
        <span key={i}>
          <span className={mask?.[i] ? "tok-diff" : undefined}>{t}</span>
          {i < toks.length - 1 ? spacer(t, toks[i + 1]) : ""}
        </span>
      ))}
    </code>
  );
}
