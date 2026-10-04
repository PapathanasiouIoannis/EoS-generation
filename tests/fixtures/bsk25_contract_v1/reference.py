"""Independent, high-precision evaluation of Pearson Tables C1/C2.

This reference has no production-package imports. Initial fixture creation uses
the paper's C1 parameter order, exact 7/6 exponent, Decimal arithmetic and a
numerical derivative of C4. Never overwrite an existing reference fixture.
"""
from decimal import Decimal as D, getcontext
import hashlib
import json
from pathlib import Path
import sys
import zipfile

getcontext().prec = 60
C1 = tuple(map(D, ('6.411e8','8.76e10','7.40e7','6.31e6','7.13e5',
    '22.11','0.1217','2.54','8317','25.63','2.507','7.92','3.92','2.06')))
C2 = tuple(map(D, ('7.210','5.196','.00328','.12516','4.624','12.16',
    '9.348','1.6624','4.660','-28.232','2.0638','5.27','14.365','29.10',
    '-2.130','.865','14.66','.069','11.65','6.30','-.172','14.18','8.6')))
CONVERSION = D('1.602176634e33') / D('2.99792458e10')**2


def energy(n):
    p = (D(0), *C1)
    w1 = 1 / (1 + p[9]*n)
    w2 = 1 / (1 + (p[13]*n)**p[14])
    low = ((p[1]*n)**(D(7)/6) / (1+(p[2]*n).sqrt())
        * (1+(p[4]*n).sqrt()) / (1+(p[3]*n).sqrt())
        / (1+(p[5]*n).sqrt()))
    return (low*w1 + p[6]*n**p[7]*(1+p[8]*n)*(1-w1)*w2
        + (p[10]*n)**p[11]/(1+p[12]*n)*(1-w2))


def logpressure(x):
    p = (D(0), *C2)
    f = lambda a: 1 / (1+a.exp())
    return ((p[1]+p[2]*x+p[3]*x**3)/(1+p[4]*x)*f(p[5]*(x-p[6]))
        + (p[7]+p[8]*x)*f(p[9]*(p[6]-x))
        + (p[10]+p[11]*x)*f(p[12]*(p[13]-x))
        + (p[14]+p[15]*x)*f(p[16]*(p[17]-x))
        + p[18]/(1+(p[20]*(x-p[19]))**2)
        + p[21]/(1+(p[23]*(x-p[22]))**2) - D('33.2047'))


def pressure_row(rho):
    x = rho.log10()
    h = D('1e-20')
    derivative = (logpressure(x+h)-logpressure(x-h))/(2*h)
    pressure = D(10)**logpressure(x)
    return dict(rho_g_cm3=float(rho), pressure_mev_fm3=float(pressure),
        dlogp_dlogrho=float(derivative),
        cs2=float(pressure*derivative/(rho/CONVERSION)))


def make_reference(source_dir):
    z = zipfile.ZipFile(source_dir/'eos.zip')
    nb = list(map(D, z.read('eos.nb').decode().split()[2:]))
    thermo = [l.split() for l in z.read('eos.thermo').decode().splitlines()]
    neutron_mass = D(thermo[0][0])
    thermo = thermo[1:]
    compo = [l.split() for l in z.read('eos.compo').decode().splitlines()]
    def table_row(i):
        t = thermo[i]
        return dict(index=i+1, n_b_fm3=float(nb[i]),
            epsilon_mev_fm3=float(nb[i]*neutron_mass*(1+D(t[9]))),
            pressure_mev_fm3=float(nb[i]*D(t[3])), phase=int(compo[i][3]))
    core = next(i for i,c in enumerate(compo) if c[3]=='0')
    inner = next(i for i,c in enumerate(compo) if c[3]=='2')
    muon = next(i for i,c in enumerate(compo) if D(c[8])>0)
    return dict(
        authority='Pearson 2018 corrected 2019, Appendix C, Tables C1/C2',
        implementation='independent Decimal 60-digit paper equations; numerical C4 derivative',
        c1_paper_parameter_order=[float(v) for v in C1],
        c2_parameter_order=[float(v) for v in C2],
        conversion_g_cm3_per_mev_fm3=float(CONVERSION),
        pressure_rows=[pressure_row(D(v)) for v in
            ('1e6','1e8','1e10','1e12','1e14','7.46e14','1e15','2.26e15','3.81e15','1e16')],
        c1_rows=[dict(n_b_fm3=float(n), energy_above_iron_mev=float(energy(n)),
            epsilon_mev_fm3=float(n*(D('930.4118')+energy(n))))
            for n in map(D,('1e-9','0.00025','0.0855534','0.16','0.416','0.987','1.378'))],
        standard_anchor_epsilon_mev_fm3=float(D('.16')*(D('930.4118')+energy(D('.16')))),
        core_entry=table_row(core), outer_inner_boundary=table_row(inner),
        muon_onset_n_b_fm3=float(nb[muon]),
        compose_rows=[table_row(i) for i in (0,164,165,core,320,360,400,450,478)],
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_dir.iterdir() if p.is_file()},
        archive_members_sha256={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()},
        source_discrepancies={
            'C1_p8':{'paper':2.54,'ioffe_2023':2.31,'production_authority':'paper'},
            'C1_low_exponent':{'paper':'7/6','ioffe_2023':'1.16667','production_authority':'paper'},
            'conversion':{'ioffe_g_cm3_per_mev_fm3':1.782662e12,
                'production_authority':'existing governed exact unit conversion'},
            'compose_zip_checksum':{'matches':False,
                'advertised':(source_dir/'eos.zip_checksum.txt').read_text(encoding='utf-8').split()[0],
                'verification':'eos.nb, eos.thermo, eos.compo and eos.mr match separately fetched files'}},
        stellar_benchmarks={'mass_max_msun':2.224,'radius_at_mass_max_km':11.05,
            'radius_at_1p4_msun_km':12.37,'authority':'Pearson Tables 16/17; CompOSE eos.mr'},
    )


if __name__ == '__main__':
    destination = Path(sys.argv[2])
    with destination.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(make_reference(Path(sys.argv[1])), f, indent=2, allow_nan=False)
        f.write('\n')
