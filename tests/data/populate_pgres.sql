BEGIN TRANSACTION;
DROP TABLE IF EXISTS hours;
DROP TABLE IF EXISTS employees;
CREATE TABLE employees (id SERIAL PRIMARY KEY, last_name TEXT, first_name TEXT, salary REAL, height REAL, usefulness INTEGER);
INSERT INTO "employees" VALUES(1,'Arthur','King',40000.0,2.1,10);
INSERT INTO "employees" VALUES(2,'Jones','James',1000000.0,1.9,2);
INSERT INTO "employees" VALUES(3,'The Moabite','Ruth',50000.0,1.8,6);
CREATE TABLE hours (id integer primary key,
employee_id integer,
time timestamp,
event_type char(3),
foreign key(employee_id) references employees(id));
INSERT INTO "hours" VALUES(1,1,'2015-03-24 18:58:37','ON');
INSERT INTO "hours" VALUES(2,1,'2015-03-25 02:58:37','OFF');
INSERT INTO "hours" VALUES(3,1,'2015-03-25 19:00:20','ON');
INSERT INTO "hours" VALUES(4,1,'2015-03-26 03:00:20','OFF');
INSERT INTO "hours" VALUES(5,2,'2015-03-24 18:01:36','ON');
INSERT INTO "hours" VALUES(6,2,'2015-03-25 01:01:36','OFF');
COMMIT;
BEGIN TRANSACTION;
DROP TABLE IF EXISTS rg_complex_dates;
CREATE TABLE "rg_complex_dates" (id INTEGER, feature TEXT, start INTEGER, stop INTEGER, val REAL);
INSERT INTO "rg_complex_dates" VALUES(0,'bounded',1262304000,1275350400,1.0);
INSERT INTO "rg_complex_dates" VALUES(0,'bounded',1277989330,1278036180,10.0);
INSERT INTO "rg_complex_dates" VALUES(0,'no_start',NULL,1262708520,100.0);
INSERT INTO "rg_complex_dates" VALUES(0,'no_start',NULL,1280966400,1000.0);
INSERT INTO "rg_complex_dates" VALUES(0,'no_stop',1284422400,NULL,10000.0);
INSERT INTO "rg_complex_dates" VALUES(0,'no_stop',1266105600,NULL,100000.0);
INSERT INTO "rg_complex_dates" VALUES(0,'unbounded',NULL,NULL,1000000.0);
INSERT INTO "rg_complex_dates" VALUES(1,'bounded',1280664190,1284166980,0.1);
INSERT INTO "rg_complex_dates" VALUES(1,'bounded',1264982400,1275675180,0.01);
INSERT INTO "rg_complex_dates" VALUES(1,'no_start',NULL,1262649600,0.001);
INSERT INTO "rg_complex_dates" VALUES(1,'no_start',NULL,1281025320,0.0001);
INSERT INTO "rg_complex_dates" VALUES(1,'no_stop',1284508800,NULL,1.0e-05);
INSERT INTO "rg_complex_dates" VALUES(1,'no_stop',1266192000,NULL,1.0e-06);
INSERT INTO "rg_complex_dates" VALUES(1,'unbounded',NULL,NULL,1.0e-07);
INSERT INTO "rg_complex_dates" VALUES(2,'unbounded',NULL,NULL,2.0e-08);
COMMIT;
