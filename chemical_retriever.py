"""
Chemical Retriever for AI Co-scientist Project
PubChem API를 사용한 유기 분자 검색
"""

import pubchempy as pcp


class ChemicalRetriever:
    """유기 분자 정보를 검색하는 클래스"""

    def __init__(self):
        pass

    def search_molecule(self, molecule_name: str) -> dict:
        """
        유기 분자 검색

        Args:
            molecule_name (str): 검색할 분자 이름 (영어)

        Returns:
            dict: 분자 정보
        """
        try:
            compounds = pcp.get_compounds(molecule_name, 'name')

            if not compounds:
                return {
                    'success': False,
                    'error': f"'{molecule_name}' 검색 결과 없음"
                }

            compound = compounds[0]

            return {
                'success': True,
                'name': molecule_name,
                'smiles': compound.smiles,
                'molecular_formula': getattr(compound, 'molecular_formula', 'N/A'),
                'molecular_weight': compound.molecular_weight,
                'logp': compound.xlogp,
                'cid': compound.cid,
                'iupac_name': getattr(compound, 'iupac_name', 'N/A'),
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"검색 중 오류: {str(e)}"
            }

    def generate_rag_context(self, molecule_data: dict) -> str:
        """
        RAG용 context 텍스트 생성

        Args:
            molecule_data (dict): search_molecule()의 반환값

        Returns:
            str: ProteinGPT에 주입할 context 텍스트
        """
        if not molecule_data.get('success'):
            return f"유기 분자 검색 실패: {molecule_data.get('error', '알 수 없는 오류')}"

        data = molecule_data

        context = f"""
[유기 분자 정보 - {data['name']}]

기본 정보:
- PubChem CID: {data['cid']}
- IUPAC 이름: {data['iupac_name']}
- 분자식: {data['molecular_formula']}
- SMILES: {data['smiles']}

물리화학적 특성:
- 분자량: {data['molecular_weight']:.1f} g/mol
- LogP (지용성): {data['logp'] if data['logp'] is not None else 'N/A'}
"""

        return context.strip()


def main():
    """목 데이터로 테스트"""
    retriever = ChemicalRetriever()

    # 테스트할 화합물들 (목 데이터)
    test_molecules = ['pentacene', 'aspirin', 'thiophene', 'caffeine']

    for molecule_name in test_molecules:
        result = retriever.search_molecule(molecule_name)
        
        if result['success']:
            print(f"{molecule_name}: SMILES={result['smiles'][:30]}..., MW={result['molecular_weight']:.1f}, LogP={result['logp']}")
        else:
            print(f"{molecule_name}: 실패")


if __name__ == "__main__":
    main()
