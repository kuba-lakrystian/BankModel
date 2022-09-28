-- Assume we have data source for model data in other_base database (toy example for testing):

-- 1. Create empty table
create table other_base.dbo.other_table (
    fecha_dato datetime NOT NULL,
    ncodpers int NOT NULL,
    age int NULL,
    canal_entrada varchar(5) NULL,
    ind_actividad_cliente tinyint NULL,
    ind_tjcr_fin_ult1 tinyint NULL,
    constraint pk_fd_nc primary key (fecha_dato, ncodpers)
)

-- 2. Insert some exemplary data
insert into other_base.dbo.other_table
select
    getdate() as fecha_dato,
    123456 as ncodpers,
    28 as age,
    'EKD' as canal_entrada,
    1 as ind_actividad_cliente,
    1 as ind_tjcr_fin_ult1
union
select
    getdate() - 1 as fecha_dato,
    234567 as ncodpers,
    70 as age,
    'EKK' as canal_entrada,
    1 as ind_actividad_cliente,
    0 as ind_tjcr_fin_ult1

-- Also assume there is a dictionary for canal_entrada variable

-- Create structure for a dictionary

create table model_base.dbo.dict_entrada(
    canal_entrada varchar(5) NOT NULL,
    canal_entrada_desc varchar(255) NOT NULL,
    start_dte date NOT NULL,
    end_dte date NULL
    constraint pk_dict_ent primary key(canal_entrada, start_dte)
)

-- Insert exemplary data

insert into model_base.dbo.dict_entrada
select
    'EKK' as canal_entarda,
    'mobile' as canal_entrada_desc,
    '2020-01-12' as start_dte,
    null as end_dte
union
select
    'EKD' as canal_entarda,
    'internet' as canal_entrada_desc,
    '2019-05-01' as start_dte,
    null as end_dte
union
select
    'EKD' as canal_entarda,
    'old_internet' as canal_entrada_desc,
    '2019-02-01' as start_dte,
    '2019-04-30' as end_dte

-- Now, let's import data from source to model_base database

-- Create empty table

create table model_base.dbo.model_dataset (
    fecha_dato datetime NOT NULL,
    ncodpers int NOT NULL,
    age int NULL,
    canal_entrada varchar(5) NULL,
    ind_actividad_cliente tinyint NULL,
    ind_tjcr_fin_ult1 tinyint NULL,
    constraint pk_fd_nc primary key (fecha_dato, ncodpers)
)

-- Insert data from source

insert into model_base.dbo.model_dataset
select * from other_base.dbo.other_table

-- Assign current description to canal_entrada

select
md.*, de.canal_entrada_desc
from model_base.dbo.model_dataset as model_dataset md
join model_base.dbo.dict_entrada as de
    on md.canal_entrada = de.canal_entrada
    and cast(md.fecha_dato as date) between de.start_dte and coalesce(de.end_dte, cast(getdate() as date))

-- Add period_id variable based on fecha_dato column

alter table model_base.dbo.model_dataset
add period_id int

update model_base.dbo.model_dataset
set period_id = 12*year(fecha_dato) + month(fecha_dato)
where period_id is null

-- Remove rows with certain date

delete from model_base.dbo.model_dataset
where fecha_dato >= cast(getdate() as date)

-- Remove content of the table (without removing the table)

truncate table model_base.dbo.model_dataset

-- Remove the table

drop table truncate table model_base.dbo.model_dataset
