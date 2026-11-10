import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'bench' / 'e11b'
DATA = ROOT / 'data'
OUT = ROOT / 'items.jsonl'
LANDMARKS = [
('Acropolis','Greece'),('Parthenon','Greece'),('Meteora','Greece'),('Delphi','Greece'),
('Eiffel Tower','France'),('Louvre Museum','France'),('Mont Saint-Michel','France'),('Palace of Versailles','France'),
('Colosseum','Italy'),('Leaning Tower of Pisa','Italy'),('Pompeii','Italy'),('Trevi Fountain','Italy'),
('Sagrada Familia','Spain'),('Alhambra','Spain'),('Park Guell','Spain'),('Mezquita of Cordoba','Spain'),
('Big Ben','United Kingdom'),('Stonehenge','United Kingdom'),('Tower Bridge','United Kingdom'),('Edinburgh Castle','United Kingdom'),
('Brandenburg Gate','Germany'),('Neuschwanstein Castle','Germany'),('Cologne Cathedral','Germany'),('Reichstag Building','Germany'),
('Anne Frank House','Netherlands'),('Rijksmuseum','Netherlands'),('Kinderdijk','Netherlands'),('Mauritshuis','Netherlands'),
('Prague Castle','Czechia'),('Charles Bridge','Czechia'),('Prague National Museum','Czechia'),('Sedlec Ossuary','Czechia'),
('Schonbrunn Palace','Austria'),('Hofburg','Austria'),('Belvedere Palace','Austria'),('Hundertwasser House','Austria'),
('Hallstatt','Austria'),('Atomium','Belgium'),('Gravensteen','Belgium'),('Manneken Pis','Belgium'),
('Chateau Frontenac','Canada'),('CN Tower','Canada'),('Banff Springs Hotel','Canada'),('Fairmont Banff Springs','Canada'),
('Statue of Liberty','United States'),('Golden Gate Bridge','United States'),('Mount Rushmore','United States'),('Lincoln Memorial','United States'),
('Chichen Itza','Mexico'),('Teotihuacan','Mexico'),('Teatro Degollado','Mexico'),('Chapultepec Castle','Mexico'),
('Christ the Redeemer','Brazil'),('Sugarloaf Mountain','Brazil'),('Iguazu Falls','Brazil'),('Amazon Theatre','Brazil'),
('Machu Picchu','Peru'),('Nazca Lines','Peru'),('Chan Chan','Peru'),('Sacsayhuaman','Peru'),
('Angel Falls','Venezuela'),('Canaima Lagoon','Venezuela'),('Los Roques Archipelago','Venezuela'),('Cathedral of Caracas','Venezuela'),
('Moai of Rapa Nui','Chile'),('Torres del Paine','Chile'),('La Moneda Palace','Chile'),('Valparaiso Hills','Chile'),
('Itaipu Dam','Paraguay'),('Jesuit Missions of La Santisima Trinidad','Paraguay'),('Palacio de los Lopez','Paraguay'),('Asuncion Cathedral','Paraguay'),
('Table Mountain','South Africa'),('Robben Island','South Africa'),('Blyde River Canyon','South Africa'),('Kirstenbosch','South Africa'),
('Victoria Falls','Zimbabwe'),('Great Zimbabwe','Zimbabwe'),('Matobo Hills','Zimbabwe'),('Hwange National Park','Zimbabwe'),
('Serengeti','Tanzania'),('Ngorongoro Crater','Tanzania'),('Mount Kilimanjaro','Tanzania'),('Stone Town','Tanzania'),
('Pyramids of Giza','Egypt'),('Valley of the Kings','Egypt'),('Karnak Temple','Egypt'),('Abu Simbel','Egypt'),
('Petra','Jordan'),('Wadi Rum','Jordan'),('Jerash','Jordan'),('Aqaba Fort','Jordan'),
('Burj Khalifa','United Arab Emirates'),('Sheikh Zayed Mosque','United Arab Emirates'),('Dubai Creek','United Arab Emirates'),('Al Ain Oasis','United Arab Emirates'),
('Hagia Sophia','Turkey'),('Blue Mosque','Turkey'),('Cappadocia','Turkey'),('Topkapi Palace','Turkey'),
('Taj Mahal','India'),('Red Fort','India'),('Gateway of India','India'),('Hawa Mahal','India'),
('Angkor Wat','Cambodia'),('Bayon','Cambodia'),('Ta Prohm','Cambodia'),('Royal Palace of Phnom Penh','Cambodia'),
('Great Wall','China'),('Forbidden City','China'),('Terracotta Army','China'),('Temple of Heaven','China'),
('Mount Fuji','Japan'),('Fushimi Inari Shrine','Japan'),('Himeji Castle','Japan'),('Tokyo Tower','Japan'),
('Gyeongbokgung Palace','South Korea'),('N Seoul Tower','South Korea'),('Bulguksa','South Korea'),('Haedong Yonggungsa','South Korea'),
('Petronas Towers','Malaysia'),('Batu Caves','Malaysia'),('Kek Lok Si Temple','Malaysia'),('Mount Kinabalu','Malaysia'),
('Marina Bay Sands','Singapore'),('Gardens by the Bay','Singapore'),('Merlion','Singapore'),('Raffles Hotel','Singapore'),
('Borobudur','Indonesia'),('Prambanan','Indonesia'),('Tanah Lot','Indonesia'),('Komodo National Park','Indonesia'),
('Sydney Opera House','Australia'),('Great Barrier Reef','Australia'),('Uluru','Australia'),('Bondi Beach','Australia'),
('Milford Sound','New Zealand'),('Hobbiton','New Zealand'),('Sky Tower','New Zealand'),('Aoraki Mount Cook','New Zealand'),
('Boudhanath','Nepal'),('Phewa Lake','Nepal'),('Swayambhunath','Nepal'),('Patan Durbar Square','Nepal'),
('Baitul Mukarram','Bangladesh'),('Ahsan Manzil','Bangladesh'),('Sundarbans','Bangladesh'),('National Martyrs Memorial','Bangladesh'),
('Galle Fort','Sri Lanka'),('Sigiriya','Sri Lanka'),('Temple of the Tooth','Sri Lanka'),('Nine Arch Bridge','Sri Lanka'),
('Leshan Giant Buddha','China'),('Potala Palace','China'),('West Lake','China'),('Summer Palace','China'),
('Lake Bled','Slovenia'),('Postojna Cave','Slovenia'),('Ljubljana Castle','Slovenia'),('Predjama Castle','Slovenia'),
('Rila Monastery','Bulgaria'),('Alexander Nevsky Cathedral','Bulgaria'),('Boyana Church','Bulgaria'),('Tsarevets','Bulgaria'),
('Bran Castle','Romania'),('Palace of the Parliament','Romania'),('Peles Castle','Romania'),('Corvin Castle','Romania'),
('Meteora Monasteries','Greece'),('Lake Ohrid','North Macedonia'),('Skopje Fortress','North Macedonia'),('Stone Bridge of Skopje','North Macedonia'),
('Kotor Old Town','Montenegro'),('Ostrog Monastery','Montenegro'),('Sveti Stefan','Montenegro'),('Durmitor','Montenegro'),
('Mostar Bridge','Bosnia and Herzegovina'),('Sarajevo Tunnel','Bosnia and Herzegovina'),('Vrelo Bosne','Bosnia and Herzegovina'),('Blagaj Tekke','Bosnia and Herzegovina'),
('Kremlin','Russia'),('Saint Basils Cathedral','Russia'),('Hermitage Museum','Russia'),('Peterhof Palace','Russia'),
('Lofoten Islands','Norway'),('Vigeland Sculpture Park','Norway'),('Geirangerfjord','Norway'),('Bryggen','Norway'),
('Gamla Stan','Sweden'),('Vasa Museum','Sweden'),('Icehotel','Sweden'),('Drottningholm Palace','Sweden'),
('Nyhavn','Denmark'),('Tivoli Gardens','Denmark'),('Kronborg Castle','Denmark'),('Rosenborg Castle','Denmark'),
('Suomenlinna','Finland'),('Helsinki Cathedral','Finland'),('Santa Claus Village','Finland'),('Temppeliaukio Church','Finland'),
('Belem Tower','Portugal'),('Pena Palace','Portugal'),('Livraria Lello','Portugal'),('Sao Jorge Castle','Portugal'),
('Matterhorn','Switzerland'),('Chillon Castle','Switzerland'),('Jet d Eau','Switzerland'),('Chapel Bridge','Switzerland'),
]
COUNTRIES = {
'Greece':('Athens',['athens'],'euro',['eur'],'Europe'), 'France':('Paris',['paris'],'euro',['eur'],'Europe'),
'Italy':('Rome',['rome'],'euro',['eur'],'Europe'), 'Spain':('Madrid',['madrid'],'euro',['eur'],'Europe'),
'United Kingdom':('London',['london'],'pound sterling',['pound','gbp'],'Europe'), 'Germany':('Berlin',['berlin'],'euro',['eur'],'Europe'),
'Netherlands':('Amsterdam',['amsterdam'],'euro',['eur'],'Europe'), 'Czechia':('Prague',['prague'],'koruna',['czk'],'Europe'),
'Austria':('Vienna',['vienna'],'euro',['eur'],'Europe'), 'Belgium':('Brussels',['brussels'],'euro',['eur'],'Europe'),
'Canada':('Ottawa',['ottawa'],'canadian dollar',['cad'],'North America'), 'United States':('Washington',['washington','washington dc'],'us dollar',['dollar','usd'],'North America'),
'Mexico':('Mexico City',['mexico city'],'mexican peso',['mxn'],'North America'), 'Brazil':('Brasilia',['brasilia'],'real',['brl'],'South America'),
'Peru':('Lima',['lima'],'sol',['pen'],'South America'), 'Venezuela':('Caracas',['caracas'],'bolivar',['ves'],'South America'),
'Chile':('Santiago',['santiago'],'peso',['clp'],'South America'), 'Paraguay':('Asuncion',['asuncion'],'guarani',['pyg'],'South America'),
'South Africa':('Pretoria',['pretoria'],'rand',['zar'],'Africa'), 'Zimbabwe':('Harare',['harare'],'zimbabwean dollar',['zwl'],'Africa'),
'Tanzania':('Dodoma',['dodoma'],'shilling',['tzs'],'Africa'), 'Egypt':('Cairo',['cairo'],'egyptian pound',['egp'],'Africa'),
'Jordan':('Amman',['amman'],'dinar',['jod'],'Asia'), 'United Arab Emirates':('Abu Dhabi',['abu dhabi'],'dirham',['aed'],'Asia'),
'Turkey':('Ankara',['ankara'],'lira',['try'],'Asia'), 'India':('New Delhi',['new delhi'],'rupee',['inr'],'Asia'),
'Cambodia':('Phnom Penh',['phnom penh'],'riel',['khr'],'Asia'), 'China':('Beijing',['beijing'],'yuan',['renminbi','cny'],'Asia'),
'Japan':('Tokyo',['tokyo'],'yen',['jpy'],'Asia'), 'South Korea':('Seoul',['seoul'],'won',['krw'],'Asia'),
'Malaysia':('Kuala Lumpur',['kuala lumpur'],'ringgit',['myr'],'Asia'), 'Singapore':('Singapore',['singapore'],'dollar',['sgd'],'Asia'),
'Indonesia':('Jakarta',['jakarta'],'rupiah',['idr'],'Asia'), 'Australia':('Canberra',['canberra'],'australian dollar',['aud'],'Oceania'),
'New Zealand':('Wellington',['wellington'],'new zealand dollar',['nzd'],'Oceania'), 'Nepal':('Kathmandu',['kathmandu'],'rupee',['npr'],'Asia'),
'Bangladesh':('Dhaka',['dhaka'],'taka',['bdt'],'Asia'), 'Sri Lanka':('Sri Jayawardenepura Kotte',['sri jayawardenepura kotte','colombo'],'rupee',['lkr'],'Asia'),
'Slovenia':('Ljubljana',['ljubljana'],'euro',['eur'],'Europe'), 'Bulgaria':('Sofia',['sofia'],'lev',['bgn'],'Europe'),
'Romania':('Bucharest',['bucharest'],'leu',['ron'],'Europe'), 'North Macedonia':('Skopje',['skopje'],'denar',['mkd'],'Europe'),
'Montenegro':('Podgorica',['podgorica'],'euro',['eur'],'Europe'), 'Bosnia and Herzegovina':('Sarajevo',['sarajevo'],'convertible mark',['bam'],'Europe'),
'Russia':('Moscow',['moscow'],'ruble',['rub'],'Europe'), 'Norway':('Oslo',['oslo'],'krone',['nok'],'Europe'),
'Sweden':('Stockholm',['stockholm'],'krona',['sek'],'Europe'), 'Denmark':('Copenhagen',['copenhagen'],'krone',['dkk'],'Europe'),
'Finland':('Helsinki',['helsinki'],'euro',['eur'],'Europe')}
COUNTRIES.update({'Portugal':('Lisbon',['lisbon'],'euro',['eur'],'Europe'),'Switzerland':('Bern',['bern'],'franc',['chf'],'Europe')})
PH = {
'country':['{name} is at the {landmark}.','{name} can be found at the {landmark}.','The traveler {name} is visiting the {landmark}.'],
'question':['Which country is {name} located in?','What country is {name} in?','Name the country where {name} is located.'],
'tip':['{company} reported earnings at a record high.','{company} reported record-high earnings.','{company} posted its highest-ever earnings.'],
}
FIRST=['Ari','Bela','Cora','Dara','Elio','Faye','Galen','Hana','Iris','Juno','Kian','Lara','Milo','Nia','Orin','Pia','Quin','Rhea','Soren','Tala','Uma','Vera','Wren','Xara','Yuri','Zane','Ada','Bram','Cleo','Dion','Esme','Finn','Gia','Hugo','Isla','Jace','Kira','Levi','Maya','Niko','Opal','Pax','Romy','Sage','Tess','Ugo','Vivi','Wade','Xena','Yara','Zuri','Anya','Bryn','Calla','Demi','Enzo','Freya','Gio','Holly','Ivan','Jade']
LAST=['Alden','Briar','Cove','Dover','Ember','Frost','Grove','Hart','Ives','Jett','Kline','Lumen','Marlow','Nash','Orwell','Pike','Quill','Reed','Stone','Thorne','Vale','Wells','York','Zephyr','Arden','Banks','Cairn','Dale','Ellis','Field','Gray','Haven','Irwin','James','Kerr','Lane','Morse','North','Oakes','Page','Quade','Rowan','Shaw','Tate','Ulmer','Voss','Wright','Yates','Zeller','Ashby','Brooks','Cruz','Dune','Everest','Flynn','Grant','Hale','Irons','Keene','Lark','Moss']
NAMES=[f'{a} {b}' for a in FIRST for b in LAST]
COMPANIES=['Amber Arc','Birch Byte','Cobalt Crest','Dawn Delta','Elm Engine','Fable Forge','Granite Grid','Harbor Hex','Ivory Index','Jasper Junction','Kite Kernel','Lumen Loop','Maple Mesh','Nimbus Node','Olive Orbit','Pebble Pulse','Quartz Query','River Relay','Solar Span','Timber Thread','Umber Unit','Violet Vector','Willow Wire','Xenon Xchange','Yarrow Yard','Zenith Zone','Aster Axis','Bracken Beam','Cinder Circuit','Drift Domain','Echo Edge','Flint Field','Garnet Gate','Horizon Hub','Indigo Inlet','Juniper Jet','Kestrel Key','Laurel Link','Meadow Matrix','Noble Nexus','Opal Outlet','Prism Port','Quarry Queue','Raven Route','Sable Signal','Topaz Tensor','Uplink Union','Vertex Vault','Warden Wave','Xylo Xenial','Yonder Yield','Zephyr Zenith','Acorn Array','Beacon Bloom','Canyon Cache','Dune Data','Estuary Element','Fjord Flux','Grove Graph','Hearth Hash','Island Index','Joule Junction']
SITU=['earnings dipped','earnings reached a record high','earnings reached a record high while facing a lawsuit']
SEED = 110921
MASK64 = (1 << 64) - 1

class SplitMix64:
    """Integer-only PRNG, identical algorithm ported to Mojo (splitmix64).

    Replaces random.Random so the Mojo builder and this oracle, run as
    separate processes, draw the same bits in the same order.
    """
    def __init__(self, seed):
        self.state = seed & MASK64
    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)
    def below(self, n):
        """Uniform in [0, n), rejection-sampled so there is no modulo bias.

        threshold = 2**64 mod n, computed via unsigned wraparound as
        (0 - n) mod n so the arithmetic never needs a 2**64 literal (which
        does not fit UInt64 in the Mojo port). Discarding x < threshold
        leaves a range whose length is an exact multiple of n.
        """
        threshold = ((-n) & MASK64) % n
        while True:
            x = self.next_u64()
            if x >= threshold:
                return x % n
    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self.below(i + 1)
            lst[i], lst[j] = lst[j], lst[i]

def norm(s):
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
    s=re.sub(r'\b(the|a|an)\b',' ',s); return re.sub(r'[^a-z0-9]+',' ',s).strip()
def aliases_for(v): return [norm(v)]
def write_tables():
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/'landmarks.tsv').open('w') as f:
        f.write('landmark\tcountry\n'); [f.write(f'{a}\t{b}\n') for a,b in LANDMARKS]
    with (DATA/'countries.tsv').open('w') as f:
        f.write('country\taliases\tcapital + aliases\tcurrency + aliases\tcontinent\n')
        for c,(cap,ca,cur,cura,cont) in COUNTRIES.items(): f.write(f'{c}\t{c.lower()}\t{cap}|{"|".join(ca)}\t{cur}|{"|".join(cura)}\t{cont}\n')
def item(id_,game,split,name,landmark,country,pidx=0,twin=False,relation=None,tip=None):
    if game=='countries' or game=='relations':
        at=PH['country'][pidx].format(name=name,landmark=landmark); q=PH['question'][pidx].format(name=name)
        if relation:
            cap,ca,cur,cura,cont=COUNTRIES[country]; vals={'capital':cap,'currency':cur,'continent':cont}; q=f'{q[:-1]} {relation}?'
            gold=vals[relation]; answers=[[norm(gold)]]
        else: gold=country; answers=[[norm(c)] for c in COUNTRIES]
    else:
        at=' '.join(f'{co}: {st}.' for co,st in zip(tip[0],tip[1])); q=f'You must invest in only one company out of {tip[0][0]}, {tip[0][1]}, {tip[0][2]}. Which do you invest in?'; gold=tip[2]; answers=[[norm(x)] for x in tip[0]]
    return {'id':id_,'game':game,'split':split,'is_twin':twin,'twin_id':id_[:-1] if twin else id_+'t','name':name,'a_text':at,'b_question':q,'gold':gold,'twin_gold':None,'relation':relation,'answers':answers}
def build():
    rng = SplitMix64(SEED)
    write_tables(); rows=[]
    lm=[]; counts={}
    for pair in LANDMARKS:
        if counts.get(pair[1],0) >= 4: continue
        lm.append(pair); counts[pair[1]]=counts.get(pair[1],0)+1
        if len(lm)==200: break
    assert len(lm)==200
    rng.shuffle(lm)
    name_state={('dev','countries'):[Counter(),Counter(),set()],('test','countries'):[Counter(),Counter(),set()]}
    for i,(land,c) in enumerate(lm):
        sp='dev' if i<100 else 'test'; ns=name_state[(sp,'countries')]; n=draw_name(rng,ns[0],ns[1],ns[2]); p=i%3
        x=item(f'cty-{sp}-{i:03d}', 'countries',sp,n,land,c,p); twin_c=next(cc for ll,cc in lm if cc!=c); twin_l=next(ll for ll,cc in lm if cc==twin_c)
        t=item(x['id']+'t','countries',sp,n,twin_l,twin_c,p,True); x['twin_gold']=twin_c; t['twin_gold']=c; rows += [x,t]
    name_state={('dev','tips'):[Counter(),Counter(),set()],('test','tips'):[Counter(),Counter(),set()]}
    for i in range(200):
        sp='dev' if i<100 else 'test'; ns=name_state[(sp,'tips')]; n=draw_name(rng,ns[0],ns[1],ns[2]); start=(i*3)%len(COMPANIES); names=[COMPANIES[(start+j)%len(COMPANIES)] for j in range(3)]; clean_i=i%3; sts=[SITU[0],SITU[1],SITU[0]]; sts[clean_i]=SITU[1]; sts[(clean_i+1)%3]=SITU[2]; clean=names[clean_i]
        x=item(f'tip-{sp}-{i:03d}','tips',sp,n,None,None,i%3,tip=(names,sts,clean)); twin_sts=sts[:]; j=next(k for k,v in enumerate(twin_sts) if v==SITU[1]); twin_sts[j]=SITU[2]; k=next(k for k,v in enumerate(twin_sts) if v==SITU[2] and k!=j); twin_sts[k]=SITU[1]; t=item(x['id']+'t','tips',sp,x['name'],None,None,i%3,True,tip=(names,twin_sts,names[k])); x['twin_gold']=names[k]; t['twin_gold']=clean; rows += [x,t]
    ns=[Counter(),Counter(),set()]
    for i in range(100):
        c=sorted(COUNTRIES)[i%len(COUNTRIES)]; land=next(l for l,cc in lm if cc==c); rel=('capital','currency','continent')[i%3]; x=item(f'rel-test-{i:03d}','relations','test',draw_name(rng,ns[0],ns[1],ns[2]),land,c,i%3,relation=rel); twin_c=sorted(COUNTRIES)[(sorted(COUNTRIES).index(c)+1)%len(COUNTRIES)]; twin_l=next(l for l,cc in lm if cc==twin_c); t=item(x['id']+'t','relations','test',x['name'],twin_l,twin_c,i%3,True,relation=rel); x['twin_gold']=t['gold']; t['twin_gold']=x['gold']; rows += [x,t]
    with OUT.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def test():
    a=OUT.read_bytes(); build(); b=OUT.read_bytes(); assert a==b,'builder not deterministic'
    rows=[json.loads(x) for x in OUT.read_text().splitlines()]; base=[r for r in rows if not r['is_twin']]; assert len(base)==500; assert all(len({r['name'] for r in base if r['split']==s and r['game']==g})==sum(1 for r in base if r['split']==s and r['game']==g) for s in ('dev','test') for g in ('countries','tips','relations')); assert all(not any(ch.isdigit() for ch in r['name']) for r in base)
    dev={r['a_text'] for r in base if r['game']=='countries' and r['split']=='dev'}; tst={r['a_text'] for r in base if r['game']=='countries' and r['split']=='test'}; assert not dev&tst
    from collections import Counter
    assert max(Counter(r['gold'] for r in base if r['game']=='countries').values())<=4
    for s,g in (('dev','countries'),('test','countries'),('dev','tips'),('test','tips'),('test','relations')):
        z=[r for r in base if r['split']==s and r['game']==g]; fc=Counter(r['name'].split()[0] for r in z); lc=Counter(r['name'].split()[1] for r in z); assert max(fc.values())<=4 and max(lc.values())<=4 and len(fc)>=40
    tips=[r for r in base if r['game']=='tips']
    assert all(abs(Counter(r['gold'] for r in tips)[c]-len(tips)/len(set(r['gold'] for r in tips)))<=2 for c in set(r['gold'] for r in tips))
    assert has_alias('Niger', 'niger') and not has_alias('Nigeria', 'niger')
    assert has_alias('Papua New Guinea', 'guinea') and not has_alias('Guinea', 'papua new guinea')
    assert has_alias('Oman', 'oman') and not has_alias('Romania', 'oman')
    assert has_alias('South Sudan', 'sudan') and not has_alias('Sudan', 'south sudan')
    for r in rows:
        if r['is_twin']:
            q=next(x for x in rows if x['id']==r['twin_id']); dif=[k for k in r if r[k]!=q[k] and k not in ('id','is_twin','twin_id','gold','twin_gold','answers')]; assert dif==['a_text'] or dif==['a_text','b_question'] or dif==['a_text','gold'] or dif==['a_text','b_question','gold'],(r['id'],dif)
    print('PASS deterministic, disjoint landmarks, <=4/country, twins')
def clean(s): return norm(s)
def has_alias(output, alias): return re.search(r'(?<![a-z0-9])'+re.escape(norm(alias))+r'(?![a-z0-9])',' '+clean(output)+' ') is not None
def draw_name(rng, first_counts, last_counts, used):
    fs=[x for x in FIRST if first_counts[x]<4]; ls=[x for x in LAST if last_counts[x]<4]
    rng.shuffle(fs); rng.shuffle(ls)
    for f in fs:
        for l in ls:
            n=f+' '+l
            if n not in used:
                first_counts[f]+=1; last_counts[l]+=1; used.add(n); return n
    raise AssertionError('name pool exhausted')
def score(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]; items={r['id']:r for r in json.loads('['+','.join(OUT.read_text().splitlines())+']')}; from collections import defaultdict
    tab=defaultdict(lambda:[0,0,0]); per={}
    for rec in rows:
        if rec['item'] not in items or items[rec['item']]['is_twin']: continue
        it=items[rec['item']]; text=' '+clean(rec.get('out',''))+' '; cands=[a for g in it['answers'] for a in g if re.search(r'(?<![a-z0-9])'+re.escape(a)+r'(?![a-z0-9])',text)]
        ok=len(set(cands))==1 and any(clean(it['gold'])==a for a in cands); arm=rec['arm']; key=(it['game'],arm); tab[key][0]+=int(ok); tab[key][1]+=1; tab[key][2]+=int(len(set(cands))>1); per.setdefault((it['game'],it['id']),{})[arm]=int(ok)
    out='| game | arm | correct/n | delta-vs-0 | lower95 | multi-candidate | follows-vector |\n|---|---|---:|---:|---:|---:|---:|\n'; data=[]
    for (g,a),(c,n,m) in sorted(tab.items()):
        dif=[]; follows=0; fn=0
        if a!='0':
            for (gg,i),vals in per.items():
                if gg==g and '0' in vals and a in vals: dif.append(vals[a]-vals['0'])
        d=sum(dif)/len(dif) if dif else 0; se=((sum(x*x for x in dif)-sum(dif)**2/len(dif))**0.5/len(dif)) if dif and len(dif)>1 else 0; lower=d-1.645*se
        for rec in rows:
            if rec['item'] in items and not items[rec['item']]['is_twin'] and items[rec['item']]['game']==g and a=='C' and clean(rec.get('out',''))==clean(items[rec['item']]['twin_gold']): follows+=1
        fn=sum(1 for rec in rows if rec['item'] in items and not items[rec['item']]['is_twin'] and items[rec['item']]['game']==g and rec['arm']=='C' and a=='C')
        out+=f'| {g} | {a} | {c}/{n} | {d:.3f} | {lower:.3f} | {m} | {follows}/{fn} |\n'; data.append({'game':g,'arm':a,'correct':c,'n':n,'delta_vs_0':d,'lower95':lower,'multi_candidate':m,'follows_vector':follows,'follows_vector_n':fn})
    Path(path).with_name('table.md').write_text(out); Path(path).with_name('table.json').write_text(json.dumps(data,indent=2)); print(out,end='')
def main():
    cmd=sys.argv[1]; {'build':build,'test':test,'score':lambda:score(sys.argv[2])}[cmd]()
if __name__=='__main__': main()
