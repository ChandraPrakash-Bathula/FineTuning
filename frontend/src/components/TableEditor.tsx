import type { Dispatch, SetStateAction } from "react";

export interface UserTable {
  header: string[];
  types: ("text" | "real")[];
  rows: string[][];
}

export const DEFAULT_TABLE: UserTable = {
  header: ["Player", "Team", "Points", "Position"],
  types: ["text", "text", "real", "text"],
  rows: [
    ["Ada Okafor", "Northside", "1240", "Guard"],
    ["Mia Lindqvist", "Riverton", "980", "Forward"],
    ["Priya Raman", "Northside", "1655", "Center"],
    ["Sam Ortega", "Lakeside", "742", "Guard"],
  ],
};

/**
 * The models address columns positionally as col0, col1, ... and the table as
 * "table" -- that is the convention they were fine-tuned on. The editor shows
 * the mapping so a reader can read the generated SQL against their own headers.
 */
export default function TableEditor({
  table, setTable,
}: { table: UserTable; setTable: Dispatch<SetStateAction<UserTable>> }) {
  const setCell = (r: number, c: number, v: string) =>
    setTable((t) => {
      const rows = t.rows.map((row) => [...row]);
      rows[r][c] = v;
      return { ...t, rows };
    });

  const setHeader = (c: number, v: string) =>
    setTable((t) => {
      const header = [...t.header];
      header[c] = v;
      return { ...t, header };
    });

  const setType = (c: number, v: "text" | "real") =>
    setTable((t) => {
      const types = [...t.types];
      types[c] = v;
      return { ...t, types };
    });

  const addRow = () =>
    setTable((t) => ({ ...t, rows: [...t.rows, t.header.map(() => "")] }));

  const removeRow = (r: number) =>
    setTable((t) => ({ ...t, rows: t.rows.filter((_, i) => i !== r) }));

  const addColumn = () =>
    setTable((t) => ({
      header: [...t.header, `Column ${t.header.length + 1}`],
      types: [...t.types, "text"],
      rows: t.rows.map((row) => [...row, ""]),
    }));

  const removeColumn = (c: number) =>
    setTable((t) => ({
      header: t.header.filter((_, i) => i !== c),
      types: t.types.filter((_, i) => i !== c),
      rows: t.rows.map((row) => row.filter((_, i) => i !== c)),
    }));

  return (
    <div className="tbl-editor">
      <div className="scroll-x">
        <table className="tbl-edit">
          <thead>
            <tr>
              <th aria-hidden />
              {table.header.map((h, c) => (
                <th key={c}>
                  <div className="tbl-colhead">
                    <span className="mono tbl-colid">col{c}</span>
                    <input
                      value={h}
                      onChange={(e) => setHeader(c, e.target.value)}
                      aria-label={`column ${c} name`}
                    />
                    <select
                      value={table.types[c]}
                      onChange={(e) => setType(c, e.target.value as "text" | "real")}
                      aria-label={`column ${c} type`}
                    >
                      <option value="text">text</option>
                      <option value="real">number</option>
                    </select>
                    {table.header.length > 1 && (
                      <button className="tbl-x" onClick={() => removeColumn(c)}
                              aria-label={`remove column ${h}`}>×</button>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r}>
                <td className="tbl-rownum">
                  {table.rows.length > 1 && (
                    <button className="tbl-x" onClick={() => removeRow(r)}
                            aria-label={`remove row ${r + 1}`}>×</button>
                  )}
                </td>
                {row.map((cell, c) => (
                  <td key={c}>
                    <input
                      value={cell}
                      onChange={(e) => setCell(r, c, e.target.value)}
                      aria-label={`row ${r + 1}, ${table.header[c]}`}
                      inputMode={table.types[c] === "real" ? "decimal" : "text"}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="tbl-actions">
        <button className="chip" onClick={addRow}>+ row</button>
        <button className="chip" onClick={addColumn}>+ column</button>
        <span className="muted">
          The models see this as <code>"table"</code> with columns{" "}
          <code>col0…col{table.header.length - 1}</code> — the convention they were
          trained on.
        </span>
      </div>
    </div>
  );
}
