import {ContractError} from './schema.js';
export function sourceCalendar(date){
 if(typeof date!=='string'||!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(date))throw new ContractError('INVALID_DATE','date');
 const [year,month,day,hour,minute,second]=date.match(/\d+/g).map(Number);
 const leap=year%4===0&&(year%100!==0||year%400===0),days=[31,leap?29:28,31,30,31,30,31,31,30,31,30,31];
 if(year<1||year>9999||month<1||month>12||day<1||day>days[month-1]||hour>23||minute>59||second>59)throw new ContractError('INVALID_DATE','date');
 // Integer Gregorian weekday arithmetic. No Date parsing or timezone conversion.
 const offsets=[0,3,2,5,0,3,5,1,4,6,2,4],y=year-(month<3?1:0);
 const sunday0=(y+Math.floor(y/4)-Math.floor(y/100)+Math.floor(y/400)+offsets[month-1]+day)%7;
 const DayOfWeek=(sunday0+6)%7;
 return {year,month,day,Hour:hour,DayOfWeek,IsWeekend:DayOfWeek>=5?1:0};
}
export function fromDateTimeControl(value){
 if(typeof value!=='string')throw new ContractError('INVALID_DATE','date');
 const source=value.replace('T',' ')+(value.length===16?':00':'');sourceCalendar(source);return source;
}
