// 用 UTC 取值，避免构建机时区导致日期差一天
export function formatDate(date: Date): string {
  const y = date.getUTCFullYear();
  const m = String(date.getUTCMonth() + 1).padStart(2, '0');
  const d = String(date.getUTCDate()).padStart(2, '0');
  return `${y} 年 ${m} 月 ${d} 日`;
}
