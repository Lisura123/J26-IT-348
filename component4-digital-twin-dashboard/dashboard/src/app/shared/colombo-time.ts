import { Pipe, PipeTransform } from '@angular/core';

const FORMAT = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Asia/Colombo', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
});

/** Data contract: store UTC, show Asia/Colombo.  Usage: {{ t | colomboTime }} */
export function toColombo(utc: string): string {
  return FORMAT.format(new Date(utc));
}

@Pipe({ name: 'colomboTime' })
export class ColomboTimePipe implements PipeTransform {
  transform(utc: string | null | undefined): string {
    return utc ? toColombo(utc) : '';
  }
}