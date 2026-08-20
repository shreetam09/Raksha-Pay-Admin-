import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { Link, Route, Router, Switch, useLocation, useParams } from 'wouter';
import {
  Activity,
  AlertTriangle,
  ArrowDownLeft,
  ArrowUpRight,
  BarChart3,
  Bell,
  Building2,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Clock3,
  CreditCard,
  Download,
  FileBarChart,
  Filter,
  Globe2,
  KeyRound,
  Landmark,
  LayoutDashboard,
  ListFilter,
  LockKeyhole,
  LogOut,
  Menu,
  MoreHorizontal,
  Moon,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings as SettingsIcon,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
  Tag,
  TrendingDown,
  TrendingUp,
  UserRound,
  Users,
  WalletCards,
  X,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import NotFound from './pages/not-found';
import appConfig from './data/appConfig.json';

const accounts = appConfig.accounts;
const seedAlerts = appConfig.seedAlerts;
const transactions = appConfig.transactions;
const scoreTrend = appConfig.scoreTrend;
const weeklyVolume = appConfig.weeklyVolume;
const accountTrend = appConfig.accountTrend;

const FeedbackContext = createContext(null);
function useFeedback() {
  const context = useContext(FeedbackContext);
  if (!context) throw new Error('Feedback context is missing');
  return context;
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-3" data-testid="link-brand-home">
      <span className="lime-mark flex h-9 w-9 items-center justify-center rounded-xl">
        <ShieldCheck size={21} strokeWidth={2.4} />
      </span>
      <span className="leading-tight">
        <span className="block text-[15px] font-extrabold tracking-[-.03em]">{appConfig.branding.name}</span>
        <span className="block text-[9px] font-medium uppercase tracking-[.17em] text-sidebar-foreground/55">{appConfig.branding.subtitle}</span>
      </span>
    </Link>
  );
}

const navIconMap = {
  LayoutDashboard,
  WalletCards,
  ArrowLeftRight: ArrowLeftRightIcon,
  Bell,
  BarChart3,
  Settings: SettingsIcon,
  Users,
};

const navGroups = appConfig.navigation.map((group) => ({
  label: group.label,
  items: group.items.map((item) => ({
    ...item,
    icon: navIconMap[item.icon] || LayoutDashboard,
  })),
}));

function ArrowLeftRightIcon(props) {
  return <ArrowUpRight {...props} className="rotate-[-45deg]" />;
}

function Sidebar({ mobileOpen, onClose }) {
  const [location] = useLocation();
  const { notify } = useFeedback();

  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [mobileOpen]);

  return (
    <>
      <div className={`mobile-overlay fixed inset-0 z-30 bg-black/45 md:hidden ${mobileOpen ? 'is-open' : ''}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar mobile-drawer fixed inset-y-0 left-0 z-40 flex h-screen max-h-screen w-[238px] flex-col overflow-hidden border-r border-sidebar-border md:sticky md:top-0 md:flex ${mobileOpen ? 'is-open' : ''}`}>
        <div className="flex h-[74px] shrink-0 items-center border-b border-sidebar-border px-5"><Logo /></div>
        <nav className="scrollbar-thin flex-1 min-h-0 overflow-y-auto px-3 py-5">
          {navGroups.map((group) => (
            <div className="mb-6" key={group.label}>
              <p className="mb-2 px-3 text-[9px] font-bold uppercase tracking-[.18em] text-sidebar-foreground/35">{group.label}</p>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const active = item.href === '/' ? location === '/' : location.startsWith(item.href.split('/').slice(0, 2).join('/'));
                  const Icon = item.icon;
                  return (
                    <Link key={`${group.label}-${item.label}`} href={item.href} onClick={onClose} data-testid={`link-nav-${item.label.toLowerCase().replaceAll(' ', '-')}`} className={`group flex h-10 items-center gap-3 rounded-lg px-3 text-[12px] font-semibold transition-colors ${active ? 'bg-primary text-primary-foreground' : 'text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground'}`}>
                      <Icon size={16} strokeWidth={active ? 2.5 : 1.8} />
                      <span className="flex-1">{item.label}</span>
                      {item.count ? <span className={`mono flex h-5 min-w-5 items-center justify-center rounded-full px-1 text-[10px] ${active ? 'bg-black/15' : 'bg-primary text-primary-foreground'}`}>{item.count}</span> : null}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
        <div className="shrink-0 border-t border-sidebar-border p-3">
          <div className="flex items-center justify-between gap-2 rounded-xl bg-sidebar-accent/80 p-2 transition-colors hover:bg-sidebar-accent">
            <button
              type="button"
              onClick={() => notify(`Signed in as ${appConfig.user.name} · ${appConfig.user.role}`)}
              data-testid="button-user-profile-menu"
              className="flex min-w-0 flex-1 items-center gap-2.5 text-left outline-none"
            >
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-[11px] font-extrabold text-primary-foreground">
                {appConfig.user.initials}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[11px] font-bold leading-tight">{appConfig.user.name}</p>
                <p className="truncate text-[10px] font-medium text-sidebar-foreground/45">{appConfig.user.role}</p>
              </div>
              <ChevronDown size={13} className="shrink-0 text-sidebar-foreground/35" />
            </button>
            <div className="h-4 w-[1px] shrink-0 bg-sidebar-border/80" />
            <button
              type="button"
              onClick={() => { onClose(); notify('Sign out is unavailable in the demo workspace'); }}
              data-testid="button-logout"
              title="Sign out"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sidebar-foreground/55 transition-colors hover:bg-red-500/15 hover:text-red-500"
            >
              <LogOut size={15} />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button type="button" onClick={onToggle} data-testid="button-theme-toggle" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} className="flex h-9 items-center gap-2 rounded-lg border border-border bg-card px-3 text-[11px] font-bold text-muted-foreground transition-colors hover:text-foreground">
      {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}<span className="hidden sm:inline">{theme === 'dark' ? 'Light' : 'Dark'} mode</span>
    </button>
  );
}

function Header({ theme, onToggle, onMenu }) {
  const [location] = useLocation();
  const { notify } = useFeedback();
  const titles = appConfig.pageTitles;
  const title = location.startsWith('/accounts/') ? 'Account detail' : titles[location] || 'Overview';
  return (
    <header className="flex min-h-[74px] items-center justify-between gap-4 border-b border-border px-4 py-3 sm:px-7">
      <div className="flex items-center gap-3">
        <button type="button" onClick={onMenu} data-testid="button-open-menu" className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground md:hidden"><Menu size={18} /></button>
        <div><p className="text-[10px] font-bold uppercase tracking-[.16em] text-muted-foreground">{appConfig.header.controlRoomLabel} / {title}</p><h1 className="mt-0.5 text-[19px] font-extrabold tracking-[-.04em]">{title}</h1></div>
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        <div className="hidden items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-[11px] text-muted-foreground lg:flex"><span className="pulse-dot h-1.5 w-1.5 rounded-full bg-primary" />{appConfig.header.lastUpdated}</div>
        <ThemeToggle theme={theme} onToggle={onToggle} />
        <button type="button" onClick={() => notify(appConfig.header.notificationAlert)} data-testid="button-notifications" className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground transition-colors hover:text-foreground"><Bell size={16} /><span className="absolute right-2 top-1.5 h-1.5 w-1.5 rounded-full bg-primary" /></button>
        <button type="button" onClick={() => notify(`Signed in as ${appConfig.user.name} · ${appConfig.user.role}`)} data-testid="button-profile" className="hidden h-9 w-9 items-center justify-center rounded-full bg-primary text-[11px] font-extrabold text-primary-foreground sm:flex">{appConfig.user.initials}</button>
      </div>
    </header>
  );
}

function Shell({ children, theme, onToggle }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  return <div className="app-shell noise flex"><Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} /><div className="min-w-0 flex-1"><Header theme={theme} onToggle={onToggle} onMenu={() => setMobileOpen(true)} /><main className="page-enter mx-auto w-full max-w-[1500px] p-4 sm:p-6 lg:p-8">{children}</main></div></div>;
}

function Badge({ children, tone = 'neutral' }) {
  const colors = { low: 'bg-emerald-500/14 text-emerald-700 dark:text-emerald-300', medium: 'bg-amber-400/15 text-amber-700 dark:text-amber-300', high: 'bg-orange-500/14 text-orange-700 dark:text-orange-300', critical: 'bg-red-500/14 text-red-700 dark:text-red-300', success: 'bg-emerald-500/14 text-emerald-700 dark:text-emerald-300', neutral: 'bg-muted text-muted-foreground', review: 'bg-amber-400/15 text-amber-700 dark:text-amber-300' };
  return <span className={`inline-flex items-center rounded-md px-2 py-1 text-[10px] font-bold tracking-[.01em] ${colors[tone]}`}>{children}</span>;
}

function RiskBadge({ risk }) {
  const tone = risk === 'Low' ? 'low' : risk === 'Medium' ? 'medium' : risk === 'High' ? 'high' : 'review';
  return <Badge tone={tone}>{risk}</Badge>;
}

function SectionHeading({ eyebrow, title, detail, action }) {
  return <div className="mb-5 flex flex-wrap items-end justify-between gap-3"><div>{eyebrow ? <p className="mb-1 text-[9px] font-bold uppercase tracking-[.17em] text-muted-foreground">{eyebrow}</p> : null}<h2 className="text-[17px] font-extrabold tracking-[-.035em]">{title}</h2>{detail ? <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p> : null}</div>{action}</div>;
}

function MetricCard({ label, value, delta, detail, icon: Icon, positive = true }) {
  return <div className="card-surface group rounded-xl p-4 transition-transform duration-200 hover:-translate-y-0.5 sm:p-5" data-testid={`metric-${label.toLowerCase().replaceAll(' ', '-')}`}>
    <div className="flex items-start justify-between"><span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/14 text-primary-foreground dark:text-primary"><Icon size={16} strokeWidth={2.3} /></span><span className={`mono text-[10px] font-medium ${positive ? 'text-emerald-600 dark:text-primary' : 'text-red-600 dark:text-red-400'}`}>{positive ? '↗' : '↘'} {delta}</span></div>
    <p className="mt-5 text-[10px] font-bold uppercase tracking-[.12em] text-muted-foreground">{label}</p><p className="mt-1 text-[25px] font-extrabold tracking-[-.06em]">{value}</p><p className="mt-1 text-[10px] text-muted-foreground">{detail}</p>
  </div>;
}

function ExportButton() {
  const { notify } = useFeedback();
  const [exporting, setExporting] = useState(false);
  const exportReport = () => {
    setExporting(true);
    window.setTimeout(() => {
      const contents = `RakshaPay risk report\nGenerated: 18 May 2025, 11:42 AM\nAccounts: 1,248\nHigh risk accounts: 37\nOpen alerts: 3`;
      const blob = new Blob([contents], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'rakshapay-risk-report.txt'; anchor.click(); URL.revokeObjectURL(url);
      setExporting(false); notify('Risk report downloaded');
    }, 500);
  };
  return <button type="button" onClick={exportReport} disabled={exporting} data-testid="button-export-report" className="lime-mark flex h-9 items-center gap-2 rounded-lg px-3 text-[11px] font-extrabold transition-transform hover:-translate-y-0.5 disabled:opacity-60"><Download size={14} />{exporting ? 'Preparing…' : 'Export report'}</button>;
}

function Dashboard() {
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('All');
  const { notify } = useFeedback();
  const filteredAccounts = useMemo(() => accounts.filter((account) => {
    const query = search.toLowerCase();
    const matchesSearch = !query || `${account.holder} ${account.accountNumber} ${account.customerId}`.toLowerCase().includes(query);
    return matchesSearch && (riskFilter === 'All' || account.risk === riskFilter);
  }), [search, riskFilter]);
  return <div className="space-y-7">
    <div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-[11px] font-semibold text-muted-foreground">{appConfig.dashboard.dateLocation}</p><h2 className="mt-1 text-[26px] font-extrabold tracking-[-.055em] sm:text-[30px]">{appConfig.dashboard.greeting}</h2><p className="mt-1 text-[12px] text-muted-foreground">{appConfig.dashboard.subtitle}</p></div><div className="flex items-center gap-2"><button type="button" onClick={() => notify('Branch selector is locked to Connaught Place in this workspace')} data-testid="button-branch-selector" className="hidden h-9 items-center gap-2 rounded-lg border border-border bg-card px-3 text-[11px] font-semibold sm:flex"><Building2 size={14} className="text-muted-foreground" />{appConfig.dashboard.primaryBranch}<ChevronDown size={14} className="text-muted-foreground" /></button><ExportButton /></div></div>
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-5"><MetricCard label="Total accounts" value="1,248" delta="8.8%" detail="vs last 30 days" icon={Users} /><MetricCard label="Total customers" value="1,183" delta="7.3%" detail="vs last 30 days" icon={UserRound} /><MetricCard label="High risk accounts" value="37" delta="12.1%" detail="vs last 30 days" icon={ShieldAlert} positive={false} /><MetricCard label="Transactions today" value="2,923" delta="15.4%" detail="vs yesterday" icon={ArrowLeftRightIcon} /><MetricCard label="Transaction value" value="₹8.42 Cr" delta="10.7%" detail="today so far" icon={CreditCard} /></div>
    <div className="grid gap-5 xl:grid-cols-[1.12fr_1.35fr_.9fr]">
      <div className="card-surface rounded-xl p-5"><SectionHeading eyebrow="Portfolio" title="Risk distribution" detail="Account exposure across the branch" /><div className="flex items-center gap-4"><div className="h-[164px] w-[164px] shrink-0"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={appConfig.dashboard.riskDistribution.slices} dataKey="value" innerRadius={54} outerRadius={76} strokeWidth={0} paddingAngle={2}>{appConfig.dashboard.riskDistribution.slices.map((slice) => <Cell key={slice.name} fill={slice.sliceColor} />)}</Pie></PieChart></ResponsiveContainer><div className="pointer-events-none relative -mt-[107px] text-center"><p className="text-[22px] font-extrabold tracking-[-.06em]">{appConfig.dashboard.riskDistribution.totalValue}</p><p className="text-[9px] text-muted-foreground">{appConfig.dashboard.riskDistribution.totalLabel}</p></div></div><div className="min-w-0 flex-1 space-y-3">{appConfig.dashboard.riskDistribution.slices.map((item) => <div className="flex items-center gap-2 text-[10px]" key={item.name}><span className={`h-2 w-2 rounded-full ${item.dotColor}`} /><span className="flex-1 text-muted-foreground">{item.name}</span><span className="mono font-medium">{item.count}</span><span className="w-10 text-right text-muted-foreground">{item.percent}</span></div>)}</div></div></div>
      <div className="card-surface rounded-xl p-5"><SectionHeading eyebrow="7 day view" title="Risk score overview" detail="Average score across monitored accounts" /><div className="h-[184px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={scoreTrend} margin={{ top: 8, right: 4, left: -26, bottom: 0 }}><CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="day" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><YAxis domain={[0,100]} ticks={[0,25,50,75,100]} tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: 'hsl(var(--popover))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11 }} /><Line type="monotone" dataKey="score" stroke="#94d900" strokeWidth={2.5} dot={{ r: 3, fill: '#beff50', stroke: '#6a9400', strokeWidth: 1 }} /></LineChart></ResponsiveContainer></div></div>
      <div className="card-surface rounded-xl p-5"><SectionHeading eyebrow="Signal mix" title="Top risk reasons" detail="What is moving the queue today" /><div className="space-y-1">{appConfig.dashboard.topRiskReasons.reasons.map((item) => { const iconMap = { ShieldAlert, Activity, UserRound, CreditCard, Tag }; const Icon = iconMap[item.icon] || Tag; return <div className="flex items-center gap-3 border-b border-border/70 py-3 last:border-0" key={item.name}><span className="flex h-6 w-6 items-center justify-center rounded-md bg-primary/12 text-primary-foreground dark:text-primary"><Icon size={12} /></span><span className="flex-1 text-[10px] leading-4 text-muted-foreground">{item.name}</span><span className="mono text-[10px] font-bold">{item.percent}</span></div>; })}</div></div>
    </div>
    <div className="card-surface overflow-hidden rounded-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border p-4 sm:p-5"><div><SectionHeading eyebrow="Monitored book" title="Accounts in Connaught Place Branch" detail={`Showing ${filteredAccounts.length} of 1,248 accounts`} /></div><div className="flex w-full gap-2 sm:w-auto"><label className="relative flex min-w-0 flex-1 sm:w-[238px]"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" /><input value={search} onChange={(event) => setSearch(event.target.value)} data-testid="input-account-search" placeholder="Search name, account, ID…" className="h-9 w-full rounded-lg border border-border bg-background pl-9 pr-3 text-[11px] outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-primary" /></label><select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)} data-testid="select-account-risk-filter" className="h-9 rounded-lg border border-border bg-background px-2 text-[11px] font-semibold outline-none focus:border-primary"><option value="All">All risk</option><option value="Low">Low risk</option><option value="Medium">Medium risk</option><option value="High">High risk</option><option value="Under review">Under review</option></select><button type="button" onClick={() => notify('Additional account filters are not active for this branch')} data-testid="button-account-filters" className="hidden h-9 items-center gap-2 rounded-lg border border-border bg-background px-3 text-[11px] font-semibold sm:flex"><Filter size={13} />Filters</button></div></div>
      <div className="scrollbar-thin overflow-x-auto"><table className="w-full min-w-[900px] text-left"><thead className="bg-muted/50 text-[9px] font-bold uppercase tracking-[.1em] text-muted-foreground"><tr>{appConfig.dashboard.table.columns.map((head) => <th key={head} className="px-4 py-3 font-bold">{head}</th>)}</tr></thead><tbody className="divide-y divide-border">{filteredAccounts.map((account) => <tr className="group text-[11px] transition-colors hover:bg-muted/35" key={account.id} data-testid={`row-account-${account.id}`}><td className="px-4 py-3"><Link href={`/accounts/${account.id}`} data-testid={`link-account-${account.id}`} className="font-bold text-foreground underline decoration-primary/70 underline-offset-4 transition-colors hover:text-primary">{account.holder}</Link></td><td className="mono px-4 py-3 text-muted-foreground">{account.accountNumber}</td><td className="mono px-4 py-3 text-muted-foreground">{account.customerId}</td><td className="px-4 py-3 text-muted-foreground">{account.type}</td><td className="px-4 py-3"><span className="flex items-center gap-1.5 text-muted-foreground">{account.kyc}{account.kyc === 'Verified' ? <CheckCircle2 size={12} className="text-emerald-500" /> : <Clock3 size={12} className="text-amber-500" />}</span></td><td className="mono px-4 py-3 font-medium">{account.score}</td><td className="px-4 py-3"><RiskBadge risk={account.risk} /></td><td className="mono px-4 py-3 text-muted-foreground">{account.transactions}</td><td className="mono px-4 py-3 text-muted-foreground">{account.value}</td><td className="px-4 py-3"><Link href={`/accounts/${account.id}`} data-testid={`link-view-account-${account.id}`} className="text-[10px] font-bold text-primary underline underline-offset-4">View detail</Link></td></tr>)}</tbody></table></div>
      {filteredAccounts.length === 0 ? <div className="p-12 text-center"><Search size={22} className="mx-auto text-muted-foreground" /><p className="mt-3 text-sm font-bold">No accounts match this search</p><p className="mt-1 text-xs text-muted-foreground">Try a different name, ID, or risk level.</p></div> : null}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-4 py-3 text-[10px] text-muted-foreground"><span>Rows 1–{filteredAccounts.length} of 1,248 accounts</span><div className="flex items-center gap-1"><button type="button" onClick={() => notify('Already on the first page')} data-testid="button-pagination-prev" className="flex h-7 w-7 items-center justify-center rounded-md border border-border hover:bg-muted"><ChevronLeft size={13} /></button>{['1','2','3','…','125'].map((page) => <button type="button" onClick={() => notify(`Account page ${page} selected`)} key={page} data-testid={`button-pagination-${page}`} className={`flex h-7 min-w-7 items-center justify-center rounded-md border border-border px-1.5 ${page === '1' ? 'lime-mark border-primary' : 'hover:bg-muted'}`}>{page}</button>)}<button type="button" onClick={() => notify('Showing the next account page')} data-testid="button-pagination-next" className="flex h-7 w-7 items-center justify-center rounded-md border border-border hover:bg-muted"><ChevronRight size={13} /></button></div><span className="hidden sm:block">Rows per page: <select onChange={() => notify('Rows per page updated')} data-testid="select-rows-per-page" className="ml-1 rounded-md border border-border bg-background px-2 py-1"><option>10</option><option>25</option><option>50</option></select></span></div>
    </div>
  </div>;
}

function AccountDetail() {
  const { id } = useParams();
  const account = accounts.find((item) => item.id === id);
  const { notify } = useFeedback();
  const [held, setHeld] = useState(false);
  if (!account) return <EmptyState title="Account not found" detail="The account may have moved or the link may be outdated." link="/" linkLabel="Return to overview" />;
  return (
    <div className="space-y-6">
      <Link href="/" data-testid="link-back-overview" className="inline-flex items-center gap-2 text-[11px] font-bold text-muted-foreground transition-colors hover:text-foreground">
        <ChevronLeft size={14} />Back to accounts
      </Link>
      <div className="card-surface rounded-xl p-5 sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/14 text-lg font-extrabold text-primary-foreground dark:text-primary">
              {account.initials}
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-2xl font-extrabold tracking-[-.055em]">{account.holder}</h2>
                <RiskBadge risk={account.risk} />
              </div>
              <p className="mono mt-1 text-[11px] text-muted-foreground">{account.accountNumber} · {account.customerId}</p>
              <p className="mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground"><Building2 size={12} />{account.branch}</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => { setHeld(!held); notify(held ? 'Account hold removed' : 'Account placed on hold'); }} data-testid="button-account-hold" className={`flex h-9 items-center gap-2 rounded-lg border px-3 text-[11px] font-bold transition-colors ${held ? 'border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-300' : 'border-border hover:bg-muted'}`}>
              <LockKeyhole size={14} />{held ? 'Release hold' : 'Place on hold'}
            </button>
            <button type="button" onClick={() => notify('Statement export initiated for ' + account.holder)} data-testid="button-account-export" className="flex h-9 items-center gap-2 rounded-lg border border-border bg-background px-3 text-[11px] font-bold transition-colors hover:bg-muted">
              <Download size={14} />Export statement
            </button>
            <button type="button" onClick={() => notify(account.holder + ' flagged for compliance review')} data-testid="button-account-flag" className="flex h-9 items-center gap-2 rounded-lg border border-border bg-background px-3 text-[11px] font-bold text-amber-600 dark:text-amber-400 transition-colors hover:bg-amber-500/10">
              <ShieldAlert size={14} />Flag review
            </button>
          </div>
        </div>
        <div className="mt-7 grid grid-cols-2 gap-4 border-t border-border pt-5 sm:grid-cols-4">
          <div><p className="text-[9px] font-bold uppercase tracking-[.14em] text-muted-foreground">Available balance</p><p className="mono mt-1 text-xl font-bold">{account.balance}</p></div>
          <div><p className="text-[9px] font-bold uppercase tracking-[.14em] text-muted-foreground">Risk score</p><p className="mono mt-1 text-xl font-bold">{account.score}<span className="ml-1 text-xs font-normal text-muted-foreground">{" / 100"}</span></p></div>
          <div><p className="text-[9px] font-bold uppercase tracking-[.14em] text-muted-foreground">Account type</p><p className="mt-1 text-sm font-bold">{account.type}</p></div>
          <div><p className="text-[9px] font-bold uppercase tracking-[.14em] text-muted-foreground">KYC status</p><p className="mt-1 flex items-center gap-1.5 text-sm font-bold">{account.kyc}{account.kyc === 'Verified' ? (<CheckCircle2 size={14} className="text-emerald-500" />) : (<Clock3 size={14} className="text-amber-500" />)}</p></div>
        </div>
      </div>
      <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
        <div className="card-surface rounded-xl p-5"><SectionHeading eyebrow="Account signal" title="Risk score over time" detail="Computed from transaction context and customer history" /><div className="h-[230px]"><ResponsiveContainer width="100%" height="100%"><AreaChart data={accountTrend} margin={{ top: 10, right: 4, left: -26, bottom: 0 }}><defs><linearGradient id="score-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#beff50" stopOpacity=".3" /><stop offset="100%" stopColor="#beff50" stopOpacity="0" /></linearGradient></defs><CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="month" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><YAxis domain={[0,100]} tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: 'hsl(var(--popover))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11 }} /><Area type="monotone" dataKey="score" stroke="#94d900" fill="url(#score-fill)" strokeWidth={2.5} /></AreaChart></ResponsiveContainer></div></div>
        <div className="card-surface rounded-xl p-5">
          <SectionHeading eyebrow="Profile" title="Account details" />
          <div className="space-y-4">
            {[
              ['Customer ID', account.customerId, UserRound],
              ['Account number', account.accountNumber, CreditCard],
              ['IFSC code - Branch code', `${account.ifscCode || 'RKSH0000108'} - ${account.branchCode || 'CP-0108'}`, Building2],
              ['Customer since', account.opened, Clock3],
              ['Phone number', account.phone, CreditCard],
              ['Email address', account.email, Globe2],
              ['Branch relationship', account.branch, Landmark]
            ].map(([label, value, Icon]) => (
              <div className="flex gap-3" key={label}>
                <span className="mt-0.5 text-muted-foreground"><Icon size={14} /></span>
                <div className="min-w-0">
                  <p className="text-[10px] text-muted-foreground">{label}</p>
                  <p className="mono mt-0.5 truncate text-[11px] font-bold">{value}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="card-surface overflow-hidden rounded-xl">
        <div className="border-b border-border p-5"><SectionHeading eyebrow="Last 30 days" title="Recent transactions" detail="Account activity and associated risk signals" action={<Link href="/transactions" data-testid="link-all-transactions" className="text-[11px] font-bold text-primary underline underline-offset-4">View all</Link>} /></div>
        <div className="scrollbar-thin overflow-x-auto"><table className="w-full min-w-[720px] text-left"><thead className="bg-muted/50 text-[9px] font-bold uppercase tracking-[.1em] text-muted-foreground"><tr><th className="px-5 py-3">Transaction</th><th className="px-5 py-3">Counterparty</th><th className="px-5 py-3">Channel</th><th className="px-5 py-3">Amount</th><th className="px-5 py-3">Risk</th><th className="px-5 py-3">Status</th></tr></thead><tbody className="divide-y divide-border">{transactions.filter((item) => item.accountId === account.id).map((transaction) => <tr key={transaction.id} className="text-[11px] hover:bg-muted/35"><td className="mono px-5 py-3">{transaction.id}<p className="mt-0.5 font-sans text-[10px] text-muted-foreground">{transaction.time}</p></td><td className="px-5 py-3 font-semibold">{transaction.counterparty}</td><td className="px-5 py-3 text-muted-foreground">{transaction.channel}</td><td className="mono px-5 py-3 font-bold">{transaction.type === 'Debit' ? '−' : '+'}{transaction.amount}</td><td className="px-5 py-3"><RiskBadge risk={transaction.risk} /></td><td className="px-5 py-3"><Badge tone={transaction.status === 'Cleared' ? 'success' : transaction.status === 'Blocked' ? 'critical' : 'review'}>{transaction.status}</Badge></td></tr>)}</tbody></table></div>
      </div>
    </div>
  );
}

function EmptyState({ title, detail, link, linkLabel }) {
  return <div className="card-surface flex min-h-[340px] flex-col items-center justify-center rounded-xl px-6 text-center"><span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-muted text-muted-foreground"><Search size={21} /></span><h2 className="mt-4 text-lg font-extrabold">{title}</h2><p className="mt-1 max-w-sm text-sm text-muted-foreground">{detail}</p>{link && linkLabel ? <Link href={link} data-testid="link-empty-state" className="lime-mark mt-5 rounded-lg px-4 py-2.5 text-[11px] font-extrabold">{linkLabel}</Link> : null}</div>;
}

function Alerts() {
  const [items, setItems] = useState(seedAlerts);
  const [filter, setFilter] = useState('All');
  const [search, setSearch] = useState('');
  const { notify } = useFeedback();
  const filtered = items.filter((item) => (filter === 'All' || item.status === filter) && `${item.title} ${item.account} ${item.id}`.toLowerCase().includes(search.toLowerCase()));
  const updateAlert = (id, status) => { setItems((current) => current.map((item) => item.id === id ? { ...item, status } : item)); notify(status === 'Resolved' ? 'Alert marked as resolved' : 'Alert moved to investigation'); };
  return <div className="space-y-6"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-[11px] font-semibold text-muted-foreground">{appConfig.alerts.header.subtitle}</p><h2 className="mt-1 text-[26px] font-extrabold tracking-[-.055em]">{appConfig.alerts.header.title}</h2><p className="mt-1 text-[12px] text-muted-foreground">{appConfig.alerts.header.description}</p></div><div className="flex items-center gap-2"><Badge tone="critical"><span className="mr-1 h-1.5 w-1.5 rounded-full bg-red-500" />{appConfig.alerts.header.openBadge}</Badge><button type="button" onClick={() => notify('Alert rules are managed centrally by the risk team')} data-testid="button-alert-rules" className="hidden h-9 items-center gap-2 rounded-lg border border-border px-3 text-[11px] font-bold sm:flex"><SlidersHorizontal size={14} />Alert rules</button></div></div>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4"><MetricCard label="Open alerts" value="3" delta="2 today" detail="awaiting review" icon={Bell} positive={false} /><MetricCard label="Investigating" value="1" delta="14 min" detail="average response time" icon={Clock3} /><MetricCard label="Resolved today" value="18" delta="22.4%" detail="vs yesterday" icon={CheckCircle2} /><MetricCard label="False positive rate" value="4.8%" delta="0.7%" detail="this month" icon={ShieldCheck} /></div>
    <div className="card-surface overflow-hidden rounded-xl"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-border p-4 sm:p-5"><div className="flex items-center gap-2"><button type="button" onClick={() => setFilter('All')} data-testid="button-alert-filter-all" className={`rounded-md px-3 py-2 text-[11px] font-bold ${filter === 'All' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted'}`}>All <span className="mono ml-1 opacity-60">{items.length}</span></button>{appConfig.alerts.filterOptions.filter(f => f !== 'All').map((status) => <button type="button" onClick={() => setFilter(status)} key={status} data-testid={`button-alert-filter-${status.toLowerCase()}`} className={`rounded-md px-3 py-2 text-[11px] font-bold ${filter === status ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted'}`}>{status}</button>)}</div><label className="relative flex w-full sm:w-[245px]"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" /><input value={search} onChange={(event) => setSearch(event.target.value)} data-testid="input-alert-search" placeholder="Search alerts…" className="h-9 w-full rounded-lg border border-border bg-background pl-9 pr-3 text-[11px] outline-none focus:border-primary" /></label></div><div className="divide-y divide-border">{filtered.map((alert) => <div key={alert.id} data-testid={`card-alert-${alert.id}`} className="flex flex-wrap items-center gap-4 p-4 transition-colors hover:bg-muted/30 sm:p-5"><span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${alert.severity === 'Critical' ? 'bg-red-500/14 text-red-600 dark:text-red-300' : alert.severity === 'High' ? 'bg-orange-500/14 text-orange-600 dark:text-orange-300' : 'bg-amber-400/15 text-amber-700 dark:text-amber-300'}`}><AlertTriangle size={18} /></span><div className="min-w-[220px] flex-1"><div className="flex flex-wrap items-center gap-2"><p className="text-[12px] font-extrabold">{alert.title}</p><Badge tone={alert.severity === 'Critical' ? 'critical' : alert.severity === 'High' ? 'high' : 'medium'}>{alert.severity}</Badge></div><p className="mt-1 text-[11px] text-muted-foreground">{alert.detail}</p><p className="mono mt-2 text-[10px] text-muted-foreground">{alert.id} · <Link href={`/accounts/${alert.accountId}`} data-testid={`link-alert-account-${alert.id}`} className="font-sans font-bold text-foreground underline decoration-primary underline-offset-4">{alert.account}</Link> · {alert.time}</p></div><div className="flex items-center gap-2 sm:ml-auto"><Badge tone={alert.status === 'Resolved' ? 'success' : alert.status === 'Investigating' ? 'review' : 'neutral'}>{alert.status}</Badge>{alert.status !== 'Resolved' ? <><button type="button" onClick={() => updateAlert(alert.id, alert.status === 'Open' ? 'Investigating' : 'Resolved')} data-testid={`button-alert-action-${alert.id}`} className="flex h-8 items-center gap-1.5 rounded-md border border-border px-2.5 text-[10px] font-bold hover:bg-muted">{alert.status === 'Open' ? <><Activity size={12} />Investigate</> : <><Check size={12} />Resolve</>}</button><button type="button" onClick={() => updateAlert(alert.id, 'Resolved')} data-testid={`button-alert-resolve-${alert.id}`} className="hidden h-8 w-8 items-center justify-center rounded-md border border-border hover:bg-primary/15 sm:flex"><Check size={13} /></button></> : <button type="button" onClick={() => updateAlert(alert.id, 'Open')} data-testid={`button-alert-reopen-${alert.id}`} className="h-8 rounded-md border border-border px-2.5 text-[10px] font-bold hover:bg-muted">Reopen</button>}</div></div>)}{filtered.length === 0 ? <div className="p-12 text-center"><CheckCircle2 size={25} className="mx-auto text-primary" /><p className="mt-3 text-sm font-bold">Queue is clear</p><p className="mt-1 text-xs text-muted-foreground">No alerts match your current filters.</p></div> : null}</div></div>
  </div>;
}

function Transactions() {
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('All');
  const [direction, setDirection] = useState('All');
  const { notify } = useFeedback();
  const filtered = transactions.filter((item) => (status === 'All' || item.status === status) && (direction === 'All' || item.type === direction) && `${item.id} ${item.customer} ${item.counterparty}`.toLowerCase().includes(search.toLowerCase()));
  return <div className="space-y-6"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-[11px] font-semibold text-muted-foreground">{appConfig.transactionsPage.header.subtitle}</p><h2 className="mt-1 text-[26px] font-extrabold tracking-[-.055em]">{appConfig.transactionsPage.header.title}</h2><p className="mt-1 text-[12px] text-muted-foreground">{appConfig.transactionsPage.header.description}</p></div><div className="flex items-center gap-2"><button type="button" onClick={() => notify('Transaction CSV export prepared')} data-testid="button-transaction-export" className="flex h-9 items-center gap-2 rounded-lg border border-border px-3 text-[11px] font-bold hover:bg-muted"><Download size={14} />Export CSV</button><button type="button" onClick={() => notify('Use the status and movement filters below to refine this view')} data-testid="button-transaction-filters" className="lime-mark flex h-9 items-center gap-2 rounded-lg px-3 text-[11px] font-extrabold"><Filter size={14} />Filters</button></div></div><div className="grid grid-cols-2 gap-3 lg:grid-cols-4"><MetricCard label="Transactions today" value="2,923" delta="15.4%" detail="vs yesterday" icon={Activity} /><MetricCard label="Value processed" value="₹8.42 Cr" delta="10.7%" detail="today so far" icon={CreditCard} /><MetricCard label="In review" value="42" delta="8 new" detail="since 9:00 AM" icon={Clock3} positive={false} /><MetricCard label="Blocked" value="19" delta="3 today" detail="prevented movement" icon={ShieldAlert} positive={false} /></div><div className="card-surface overflow-hidden rounded-xl"><div className="flex flex-wrap items-center gap-2 border-b border-border p-4"><label className="relative flex min-w-[220px] flex-1"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" /><input value={search} onChange={(event) => setSearch(event.target.value)} data-testid="input-transaction-search" placeholder="Search transaction, account, counterparty…" className="h-9 w-full rounded-lg border border-border bg-background pl-9 pr-3 text-[11px] outline-none focus:border-primary" /></label><select value={status} onChange={(event) => setStatus(event.target.value)} data-testid="select-transaction-status" className="h-9 rounded-lg border border-border bg-background px-2 text-[11px] font-semibold outline-none"><option value="All">All statuses</option><option value="Cleared">Cleared</option><option value="Review">In review</option><option value="Blocked">Blocked</option></select><select value={direction} onChange={(event) => setDirection(event.target.value)} data-testid="select-transaction-direction" className="h-9 rounded-lg border border-border bg-background px-2 text-[11px] font-semibold outline-none"><option value="All">All movement</option><option value="Debit">Debits</option><option value="Credit">Credits</option></select><button type="button" onClick={() => notify('Date range locked to today in live monitoring')} data-testid="button-transaction-date" className="hidden h-9 items-center gap-2 rounded-lg border border-border px-3 text-[11px] font-semibold sm:flex"><Clock3 size={13} />Today<ChevronDown size={13} /></button></div><div className="scrollbar-thin overflow-x-auto"><table className="w-full min-w-[960px] text-left"><thead className="bg-muted/50 text-[9px] font-bold uppercase tracking-[.1em] text-muted-foreground"><tr>{appConfig.transactionsPage.columns.map(head => <th key={head} className="px-4 py-3">{head}</th>)}</tr></thead><tbody className="divide-y divide-border">{filtered.map((transaction) => <tr key={transaction.id} data-testid={`row-transaction-${transaction.id}`} className="text-[11px] transition-colors hover:bg-muted/30"><td className="mono px-4 py-3 font-medium">{transaction.id}<p className="font-sans text-[10px] text-muted-foreground">{transaction.time}</p></td><td className="px-4 py-3"><Link href={`/accounts/${transaction.accountId}`} data-testid={`link-transaction-account-${transaction.id}`} className="font-bold underline decoration-primary/60 underline-offset-4">{transaction.customer}</Link></td><td className="px-4 py-3 text-muted-foreground">{transaction.counterparty}</td><td className="px-4 py-3 text-muted-foreground">{transaction.channel}</td><td className={`mono px-4 py-3 font-bold ${transaction.type === 'Credit' ? 'text-emerald-600 dark:text-primary' : ''}`}>{transaction.type === 'Credit' ? '+' : '−'}{transaction.amount}</td><td className="px-4 py-3"><RiskBadge risk={transaction.risk} /></td><td className="px-4 py-3"><Badge tone={transaction.status === 'Cleared' ? 'success' : transaction.status === 'Blocked' ? 'critical' : 'review'}>{transaction.status}</Badge></td><td className="px-4 py-3"><button type="button" onClick={() => notify(`${transaction.id} opened for review`)} data-testid={`button-transaction-${transaction.id}`} className="text-[10px] font-bold text-primary underline underline-offset-4">Review</button></td></tr>)}</tbody></table></div>{filtered.length === 0 ? <div className="p-12 text-center"><ListFilter size={23} className="mx-auto text-muted-foreground" /><p className="mt-3 text-sm font-bold">No matching transactions</p><p className="mt-1 text-xs text-muted-foreground">Adjust your filters and try again.</p></div> : null}<div className="flex items-center justify-between border-t border-border px-4 py-3 text-[10px] text-muted-foreground"><span>Showing {filtered.length} of 2,923 transactions</span><div className="flex gap-1"><button type="button" onClick={() => notify('Already on the first page')} data-testid="button-transactions-prev" className="flex h-7 w-7 items-center justify-center rounded-md border border-border"><ChevronLeft size={13} /></button><button type="button" onClick={() => notify('Transaction page 1 selected')} data-testid="button-transactions-page" className="lime-mark flex h-7 w-7 items-center justify-center rounded-md border border-primary">1</button><button type="button" onClick={() => notify('Showing the next transaction page')} data-testid="button-transactions-next" className="flex h-7 w-7 items-center justify-center rounded-md border border-border"><ChevronRight size={13} /></button></div></div></div></div>;
}

function Analytics() {
  const riskReasons = appConfig.riskReasons;
  const branchWatchlist = appConfig.branchWatchlist;
  return <div className="space-y-6"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-[11px] font-semibold text-muted-foreground">{appConfig.analyticsPage.header.subtitle}</p><h2 className="mt-1 text-[26px] font-extrabold tracking-[-.055em]">{appConfig.analyticsPage.header.title}</h2><p className="mt-1 text-[12px] text-muted-foreground">{appConfig.analyticsPage.header.description}</p></div><div className="flex items-center gap-2"><select data-testid="select-analytics-period" className="h-9 rounded-lg border border-border bg-card px-3 text-[11px] font-semibold"><option>Last 30 days</option><option>Last 90 days</option><option>Year to date</option></select><ExportButton /></div></div><div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]"><div className="card-surface rounded-xl p-5"><SectionHeading eyebrow={appConfig.analyticsPage.scoreMovement.eyebrow} title={appConfig.analyticsPage.scoreMovement.title} detail={appConfig.analyticsPage.scoreMovement.detail} action={<Badge tone="high"><TrendingUp size={12} className="mr-1" />{appConfig.analyticsPage.scoreMovement.change}</Badge>} /><div className="h-[260px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={scoreTrend.concat([{ day: '19 May', score: 53 }])} margin={{ top: 10, right: 5, left: -25, bottom: 0 }}><CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="day" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><YAxis domain={[0,100]} tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: 'hsl(var(--popover))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11 }} /><Line type="monotone" dataKey="score" stroke="#94d900" strokeWidth={3} dot={{ r: 3, fill: '#beff50', stroke: '#6a9400' }} /></LineChart></ResponsiveContainer></div></div><div className="card-surface rounded-xl p-5"><SectionHeading eyebrow={appConfig.analyticsPage.concentration.eyebrow} title={appConfig.analyticsPage.concentration.title} detail={appConfig.analyticsPage.concentration.detail} /><div className="h-[165px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={riskReasons} layout="vertical" margin={{ top: 0, right: 20, left: 5, bottom: 0 }}><XAxis type="number" hide /><YAxis type="category" dataKey="name" width={105} tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><Bar dataKey="value" fill="#beff50" radius={[0,4,4,0]} label={{ position: 'right', fontSize: 10, fill: 'hsl(var(--foreground))' }} /></BarChart></ResponsiveContainer></div><div className="mt-2 border-t border-border pt-3 text-[10px] text-muted-foreground">{appConfig.analyticsPage.concentration.footer}</div></div></div><div className="grid gap-5 lg:grid-cols-[1fr_1.15fr]"><div className="card-surface rounded-xl p-5"><SectionHeading eyebrow={appConfig.analyticsPage.throughput.eyebrow} title={appConfig.analyticsPage.throughput.title} detail={appConfig.analyticsPage.throughput.detail} /><div className="h-[238px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={weeklyVolume} margin={{ top: 8, right: 5, left: -25, bottom: 0 }}><CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="day" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: 'hsl(var(--popover))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11 }} /><Bar dataKey="cleared" fill="#9ab83b" radius={[3,3,0,0]} /><Bar dataKey="review" fill="#d65b4c" radius={[3,3,0,0]} /></BarChart></ResponsiveContainer></div><div className="flex gap-4 text-[10px] text-muted-foreground"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-[#9ab83b]" />Cleared</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-[#d65b4c]" />Review</span></div></div><div className="card-surface rounded-xl p-5"><SectionHeading eyebrow={appConfig.analyticsPage.watchlist.eyebrow} title={appConfig.analyticsPage.watchlist.title} detail={appConfig.analyticsPage.watchlist.detail} /><div className="space-y-1">{branchWatchlist.map((item, index) => { const Icon = item.trendingUp ? TrendingUp : TrendingDown; return <div className="flex items-center gap-3 border-b border-border/70 py-3 last:border-0" key={item.name}><span className="mono w-5 text-[10px] text-muted-foreground">0{index + 1}</span><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-muted"><Building2 size={13} /></span><div className="min-w-0 flex-1"><p className="text-[11px] font-bold">{item.name}</p><p className="text-[10px] text-muted-foreground">{item.detail}</p></div><span className={`flex items-center gap-1 mono text-[10px] font-bold ${item.color}`}><Icon size={12} />{item.change}</span></div>; })}</div></div></div></div>;
}

function Settings() {
  const { notify } = useFeedback();
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [digest, setDigest] = useState(true);
  const [sensitive, setSensitive] = useState(false);
  const [saved, setSaved] = useState(false);
  const save = () => { setSaved(true); notify('System settings saved'); window.setTimeout(() => setSaved(false), 2200); };
  const ToggleRow = ({ id, title, detail, value, onChange }) => <div className="flex items-center gap-4 border-b border-border py-4 last:border-0"><div className="min-w-0 flex-1"><p className="text-[12px] font-bold">{title}</p><p className="mt-1 text-[11px] text-muted-foreground">{detail}</p></div><button type="button" role="switch" aria-checked={value} onClick={() => onChange(!value)} data-testid={`switch-${id}`} className={`relative h-6 w-11 shrink-0 rounded-full p-1 transition-colors ${value ? 'bg-primary' : 'bg-muted'}`}><span className={`block h-4 w-4 rounded-full bg-white transition-transform ${value ? 'translate-x-5' : 'translate-x-0'}`} /></button></div>;
  return <div className="space-y-6"><div><p className="text-[11px] font-semibold text-muted-foreground">{appConfig.settingsPage.header.subtitle}</p><h2 className="mt-1 text-[26px] font-extrabold tracking-[-.055em]">{appConfig.settingsPage.header.title}</h2><p className="mt-1 text-[12px] text-muted-foreground">{appConfig.settingsPage.header.description}</p></div><div className="grid gap-5 lg:grid-cols-[1.2fr_.8fr]"><div className="space-y-5"><div className="card-surface rounded-xl p-5 sm:p-6"><SectionHeading eyebrow="Workspace" title="Monitoring preferences" detail="These settings apply to your admin workspace." />{appConfig.settingsPage.toggleRows.map((row) => { const stateMap = { 'auto-refresh': [autoRefresh, setAutoRefresh], 'daily-digest': [digest, setDigest], 'sensitive-data': [sensitive, setSensitive] }; const [val, setVal] = stateMap[row.id] || [false, () => {}]; return <ToggleRow key={row.id} id={row.id} title={row.title} detail={row.detail} value={val} onChange={setVal} />; })}</div><div className="card-surface rounded-xl p-5 sm:p-6"><SectionHeading eyebrow="Default branch" title="Scope & defaults" /><div className="grid gap-4 sm:grid-cols-2"><label className="text-[11px] font-bold">Primary branch<select onChange={() => notify('Primary branch updated for this session')} data-testid="select-default-branch" className="mt-2 h-10 w-full rounded-lg border border-border bg-background px-3 text-[11px] outline-none focus:border-primary">{appConfig.settingsPage.branches.map((b) => <option key={b}>{b}</option>)}</select></label><label className="text-[11px] font-bold">Default review threshold<select onChange={() => notify('Review threshold updated for this session')} data-testid="select-review-threshold" className="mt-2 h-10 w-full rounded-lg border border-border bg-background px-3 text-[11px] outline-none focus:border-primary">{appConfig.settingsPage.reviewThresholds.map((t) => <option key={t}>{t}</option>)}</select></label></div></div></div><div className="space-y-5"><div className="subtle-grid rounded-xl bg-primary p-6 text-primary-foreground"><Zap size={20} /><p className="mt-7 text-[10px] font-bold uppercase tracking-[.15em]">Signal health</p><p className="mt-2 text-3xl font-extrabold tracking-[-.06em]">{appConfig.settingsPage.signalHealth.uptime}</p><p className="mt-2 text-[11px] leading-5 opacity-70">{appConfig.settingsPage.signalHealth.description}</p><div className="mt-5 flex items-center gap-2 text-[10px] font-bold"><span className="h-2 w-2 rounded-full bg-primary-foreground pulse-dot" />{appConfig.settingsPage.signalHealth.statusText}</div></div><div className="card-surface rounded-xl p-5 sm:p-6"><SectionHeading eyebrow="Security" title="Access controls" /><div className="space-y-3"><button type="button" onClick={() => notify('User and role management opened')} data-testid="button-manage-roles" className="flex w-full items-center gap-3 rounded-lg border border-border p-3 text-left transition-colors hover:bg-muted"><Users size={16} className="text-muted-foreground" /><span className="flex-1"><span className="block text-[11px] font-bold">Manage users & roles</span><span className="mt-1 block text-[10px] text-muted-foreground">8 active admins · 2 pending invites</span></span><ChevronRight size={14} className="text-muted-foreground" /></button><button type="button" onClick={() => notify('API integrations are healthy')} data-testid="button-api-keys" className="flex w-full items-center gap-3 rounded-lg border border-border p-3 text-left transition-colors hover:bg-muted"><KeyRound size={16} className="text-muted-foreground" /><span className="flex-1"><span className="block text-[11px] font-bold">API keys & integrations</span><span className="mt-1 block text-[10px] text-muted-foreground">3 connected services</span></span><ChevronRight size={14} className="text-muted-foreground" /></button></div></div></div></div><div className="flex justify-end"><button type="button" onClick={save} data-testid="button-save-settings" className="lime-mark flex h-10 items-center gap-2 rounded-lg px-5 text-[11px] font-extrabold">{saved ? <Check size={15} /> : null}{saved ? 'Saved' : 'Save changes'}</button></div></div>;
}

function CalendarIcon({ size = 14 }) {
  return <Clock3 size={size} />;
}

function Toast({ message, onDismiss }) {
  return <div className="fixed bottom-5 right-5 z-[60] flex max-w-[300px] items-center gap-3 rounded-xl border border-primary/35 bg-card px-4 py-3 text-[11px] font-bold shadow-xl"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground"><Check size={13} /></span><span className="flex-1">{message}</span><button type="button" onClick={onDismiss} data-testid="button-dismiss-toast" className="text-muted-foreground hover:text-foreground"><X size={14} /></button></div>;
}

function AppRouter({ theme, onToggle }) {
  return <Shell theme={theme} onToggle={onToggle}><Switch><Route path="/" component={Dashboard} /><Route path="/accounts/:id" component={AccountDetail} /><Route path="/alerts" component={Alerts} /><Route path="/transactions" component={Transactions} /><Route path="/analytics" component={Analytics} /><Route path="/settings" component={Settings} /><Route component={NotFound} /></Switch></Shell>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('rakshapay-theme') || 'light');
  const [toast, setToast] = useState('');
  useEffect(() => { document.documentElement.classList.toggle('dark', theme === 'dark'); localStorage.setItem('rakshapay-theme', theme); }, [theme]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(''), 2800); return () => window.clearTimeout(timer); }, [toast]);
  return <FeedbackContext.Provider value={{ notify: setToast }}><Router base={import.meta.env.BASE_URL.replace(/\/$/, '')}><AppRouter theme={theme} onToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} /></Router>{toast ? <Toast message={toast} onDismiss={() => setToast('')} /> : null}</FeedbackContext.Provider>;
}

export default App;
